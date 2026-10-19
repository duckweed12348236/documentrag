"""Celery 应用与随 API 生命周期管理的 worker 子进程。

应用侧：以 Redis 作为消息代理（broker）与结果后端（backend）。

配置来源与项目其它连接项一致（``pyproject.toml`` → ``config.py``）：

- ``[tool.celery]`` 的 ``broker_url`` / ``result_url`` 显式指定连接串；
- 留空时回退为按 ``[tool.redis]`` 拼装的 ``redis://`` 连接串——与短期记忆
  缓存共用同一 Redis 实例，但键空间不重叠（Celery 用队列名与
  ``celery-task-meta-*``，短期记忆用 ``stm:checkpoint:*``）。

投递任务（须先启动 worker）::

    from plugins.celery.tasks import ingest_document

    ingest_document.delay(task_id)  # 进度与产出写回 Redis 任务记录，见 task.py

worker 侧有两种启动方式：

1. 默认随 FastAPI 生命周期自动启动 / 停止——``app.register``（lifespan）里的
   ``register_plugins()`` 会拉起 worker 子进程（:class:`CeleryWorker`），
   ``close_plugins()`` 时终止它，因此无需手动执行 ``celery worker``。子进程
   命令等价于::

       python -m celery -A plugins.celery.celery_client worker [--pool=solo]

   ``--pool=solo`` 仅在 Windows 上追加——Celery 不支持 Windows 的 prefork
   池。命令行不传 ``--loglevel``：子进程内的 ``setup_logging`` 信号处理函数
   （:func:`_take_over_logging`）接管后日志统一走 loguru，级别以
   ``config.LOG_LEVEL`` 为准。

2. 脱离 API 独立启动——Windows 上 prefork 池不可用，需指定 solo / threads 池::

       celery -A plugins.celery.celery_client worker --pool=solo

   此时由信号钩子代为注册 / 注销——启动时注册（:func:`_register_worker_plugins`），
   关闭时注销（:func:`_close_worker_plugins`）：注册时排除会拉起子进程的
   ``CeleryWorker``（其余插件在子进程里复用，Milvus 的 ``register`` 幂等，
   不会重建集合）；插件与任务协程共用 :func:`start_loop` 拉起的常驻事件循环，
   避免异步资源建在随即关闭的循环上。

日志：worker 启动时接管 Celery 自带日志配置（见 :func:`_take_over_logging`），
统一由 :mod:`logger` 输出到控制台与 ``logs/`` 下按天轮转的文件，因此日志级别
以 ``config.LOG_LEVEL`` 为准，命令行的 ``--loglevel`` 不再生效。API 拉起的
子进程额外注入环境变量 ``MAS_LOG_SUFFIX=.worker``，日志落在
``logs/<日期>.worker.log``，与 API 进程分文件写入——否则两个进程同时持有
同一个日志文件句柄，跨进程轮转（每天零点）在 Windows 上会失败。

Note:
    子进程由 :class:`CeleryWorker` 的 ``close()`` 正常终止。若 API 进程被强杀
    （如 ``taskkill /F``），worker 会变成孤儿进程，需手动清理。
"""

import asyncio
import contextlib
import os
import subprocess
import sys
import threading
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from celery import Celery
from celery.signals import (
    setup_logging,
    worker_process_init,
    worker_process_shutdown,
    worker_ready,
    worker_shutdown,
)
from tortoise import Tortoise

import config
from logger import logger, setup_logger
from plugins import Plugin, close_plugins, register_plugins
from plugins.redis import close, connect

# 导入任务侧会用到的插件模块，触发 PluginMeta 登记（plugin_classes 由此填充）。
# 只在 Celery 应用侧导入，避免每次 import plugins 都拉起这些重模块；worker
# 进程加载本模块后，即可在启动信号里统一注册它们。
import plugins.agents.embedding  # noqa: E402,F401
import plugins.agents.evaluation.graph  # noqa: E402,F401
import plugins.milvus  # noqa: E402,F401
import plugins.mineru  # noqa: E402,F401
import plugins.minio  # noqa: E402,F401

__all__ = [
    "APP_PATH",
    "CeleryWorker",
    "LOG_SUFFIX_ENV",
    "build_redis_url",
    "celery_client",
    "run_async",
    "start_loop",
    "stop_loop",
    "worker_command",
]


def build_redis_url(db: int | None = None) -> str:
    """按 :mod:`config` 中的 Redis 配置拼装连接串。

    Args:
        db: 数据库序号，缺省取 ``config.REDIS_DB``。

    Returns:
        形如 ``redis://[:password@]host:port/db`` 的连接串。
    """
    auth = f":{config.REDIS_PASSWORD}@" if config.REDIS_PASSWORD else ""
    database = config.REDIS_DB if db is None else db
    return f"redis://{auth}{config.REDIS_HOST}:{config.REDIS_PORT}/{database}"


celery_client = Celery(
    "mas",
    broker=config.CELERY_BROKER_URL or build_redis_url(),
    backend=config.CELERY_RESULT_URL or build_redis_url(),
    # 任务模块显式声明，worker 启动时自动导入，无需在命令行补 -I
    include=["plugins.celery.tasks"],
)

celery_client.conf.update(
    # 收发只走 JSON：默认的 pickle 反序列化可执行任意代码，务必禁用
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone=config.TORTOISE_TIMEZONE,
    enable_utc=True,
    task_track_started=True,
    # 转写属于耗时任务：晚确认 + 单条预取，避免 worker 重启时丢任务或堆积
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    # 父进程等待子进程就绪的时长（prefork 池透传给 billiard 的 proc_alive_timeout）。
    # 默认 4s：worker_process_init 的信号处理函数阻塞超过它，子进程会被判为启动
    # 失败并回收；插件注册要建连接、可能超过 4s，故放宽到 120s。
    worker_proc_alive_timeout=120.0,
)


@setup_logging.connect
def _take_over_logging(**kwargs: Any) -> bool:
    """接管 Celery 的日志配置，统一由 loguru 输出。

    Celery 在 ``setup_logging_subsystem`` 中检测到该信号的接收者后，会跳过
    自身的 root / task logger 配置，日志因此沿 ``propagate`` 汇入 root，由
    :mod:`logger` 的 ``InterceptHandler`` 转发给 loguru。

    Returns:
        固定为 ``True``——接收者的返回值决定 ``receivers`` 是否非空，返回真值
        才会让 Celery 跳过自带配置。
    """
    setup_logger()
    logger.debug("Celery 日志已交由 loguru 接管：{}", kwargs.get("loglevel"))
    return True


# --------------------------------------------------------------------------- #
# worker 进程：常驻事件循环 + 插件注册 / 注销信号
# --------------------------------------------------------------------------- #
#: worker 进程内插件与任务共用的常驻事件循环
_loop: asyncio.AbstractEventLoop | None = None
#: 运行上述事件循环的线程
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()
_loop_ready = threading.Event()
#: 插件是否已在本进程注册（worker_process_init / worker_ready 取先到者）
_plugins_ready = False

#: 常驻事件循环线程名
LOOP_THREAD_NAME = "celery-plugin-loop"
#: 启动 / 停止常驻事件循环的等待时长（秒）
LOOP_START_TIMEOUT = 10.0
LOOP_STOP_TIMEOUT = 10.0


def _run_loop() -> None:
    """常驻事件循环的线程体：跑 ``run_forever``，退出时收尾并关闭循环。

    与 API 进程一致，Windows 上强制用 ``SelectorEventLoop``——psycopg 的异步连接
    （短期记忆与业务库）不能在默认的 ``ProactorEventLoop`` 上运行。
    """
    global _loop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _loop = loop
    _loop_ready.set()
    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        _loop = None
        _loop_ready.clear()


def start_loop() -> asyncio.AbstractEventLoop:
    """启动（或复用）worker 进程内共用的常驻事件循环。

    httpx 异步客户端、psycopg 连接池等资源都绑定在创建它们的事件循环上，而
    Celery 的同步任务本身没有事件循环。让「插件注册」与「任务协程」共用同
    一个常驻循环，避免资源建在随即关闭的循环上、后续任务无法复用。

    Returns:
        正在运行的常驻事件循环。

    Raises:
        RuntimeError: 事件循环线程未能在 :data:`LOOP_START_TIMEOUT` 内就绪。
    """
    global _loop_thread
    with _loop_lock:
        if _loop_thread is None or not _loop_thread.is_alive():
            _loop_ready.clear()
            _loop_thread = threading.Thread(
                target=_run_loop, name=LOOP_THREAD_NAME, daemon=True
            )
            _loop_thread.start()
        if not _loop_ready.wait(timeout=LOOP_START_TIMEOUT):
            raise RuntimeError("Celery 常驻事件循环启动超时")
        loop = _loop
        if loop is None:
            raise RuntimeError("Celery 常驻事件循环启动失败")
        return loop


def stop_loop() -> None:
    """停止常驻事件循环并等待其线程退出（幂等）。"""
    global _loop_thread
    loop = _loop
    if loop is not None:
        loop.call_soon_threadsafe(loop.stop)
    if _loop_thread is not None:
        _loop_thread.join(timeout=LOOP_STOP_TIMEOUT)
        _loop_thread = None


def _shared_loop() -> asyncio.AbstractEventLoop | None:
    """返回可投递协程的常驻循环；当前不可用时返回 ``None``。

    fork 出的子进程会继承 ``_loop`` / ``_loop_thread`` 引用，但循环线程不随
    fork 复制（``threading`` 在 fork 后会把它标记为已停止），因此这里同时校验
    线程存活且非当前线程，避免把协程投递到已不存在的循环上。
    """
    loop, thread = _loop, _loop_thread
    if loop is None or thread is None or not loop.is_running():
        return None
    if not thread.is_alive() or threading.current_thread() is thread:
        return None
    return loop


@worker_process_init.connect
@worker_ready.connect
def _register_worker_plugins(**kwargs: Any) -> None:
    """worker 进程启动时注册插件。

    ``worker_process_init`` 覆盖 prefork 池的每个子进程，``worker_ready`` 覆盖
    solo / threads 池（Windows 强制 solo）；两者取先到者，重复触发由
    :data:`_plugins_ready` 幂等兜底。除 ``CeleryWorker``（否则子进程会再拉起
    一个 worker）外全部注册。

    启动过程依次建四样东西，且都在同一个常驻事件循环（:func:`start_loop`）上：
    事件循环本身、任务记录用的常驻 Redis 连接（:func:`~plugins.redis.connect`）、
    业务库连接（数据集类任务要读写 PostgreSQL）、插件。后三者各自独立 fail-open——
    任一失败只记 ERROR，不阻断 worker 消费。

    Note:
        ``worker_process_init`` 阻塞过久会被父进程判为启动失败并回收子进程，
        判定时长由配置项 ``worker_proc_alive_timeout`` 决定（默认 4s，本应用
        在 :data:`celery_client` 上已放宽到 120s）。
    """
    global _plugins_ready
    if _plugins_ready:
        return
    _plugins_ready = True
    try:
        start_loop()
    except Exception as exc:  # noqa: BLE001 - 循环起不来，注册与任务都无从谈起
        logger.exception("Celery worker 常驻事件循环启动失败：{}", exc)
        return

    # 任务记录连接：常驻本进程，任务执行时直接复用，不现建
    try:
        run_async(connect())
        logger.info("Celery worker 任务记录连接已就绪")
    except Exception as exc:  # noqa: BLE001 - 连不上不阻断 worker
        logger.exception("Celery worker 任务记录连接失败：{}", exc)

    # 业务库：数据集创建 / 评测任务要读写 PostgreSQL（与 API 进程各建一份连接池）
    try:
        run_async(Tortoise.init(config=config.TORTOISE_ORM, _enable_global_fallback=True))
        logger.info("Celery worker 数据库已初始化")
    except Exception as exc:  # noqa: BLE001 - 连不上不阻断 worker
        logger.exception("Celery worker 数据库初始化失败：{}", exc)

    try:
        run_async(register_plugins(exclude=(CeleryWorker,)))
        logger.info("Celery worker 插件已注册")
    except Exception as exc:  # noqa: BLE001 - 注册失败不阻断 worker
        logger.exception("Celery worker 插件注册失败：{}", exc)


@worker_process_shutdown.connect
@worker_shutdown.connect
def _close_worker_plugins(**kwargs: Any) -> None:
    """worker 进程关闭时注销插件、释放任务记录连接并停止常驻事件循环。

    ``worker_process_shutdown`` 覆盖 prefork 池子进程，``worker_shutdown`` 覆盖
    solo / threads 池；Celery 不保证子进程侧的该信号一定派发，因此注销与释放都按
    best-effort 处理——即使漏掉，进程退出也会回收连接。
    """
    global _plugins_ready
    if not _plugins_ready:
        return
    _plugins_ready = False

    try:
        run_async(close_plugins())
        logger.info("Celery worker 插件已注销")
    except Exception as exc:  # noqa: BLE001 - 注销失败不影响进程退出
        logger.warning("Celery worker 插件注销失败：{}", exc)

    # 常驻连接在事件循环关掉之前释放
    try:
        run_async(close())
        logger.info("Celery worker 任务记录连接已关闭")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Celery worker 任务记录连接关闭失败：{}", exc)

    try:
        run_async(Tortoise.close_connections())
        logger.info("Celery worker 数据库连接已释放")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Celery worker 数据库连接释放失败：{}", exc)

    stop_loop()


def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """在同步的 Celery 任务中驱动协程（ASR 客户端是异步实现）。

    优先投递到 worker 进程的常驻循环（:func:`start_loop`）：插件就在该循环里
    注册，任务因此复用同一份 httpx 客户端 / 连接池。循环不可用时回退——当前
    线程没有运行中的事件循环就直接 ``asyncio.run``；若已在事件循环中（例如在
    FastAPI 请求里直接调用任务函数体），则另起线程执行，避免嵌套事件循环报错。

    Args:
        coro: 待执行的协程对象。

    Returns:
        协程的返回值。
    """
    loop = _shared_loop()
    if loop is not None:
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    logger.debug("当前线程已有事件循环，改由独立线程执行协程")
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


# --------------------------------------------------------------------------- #
# 子进程：随 API 生命周期启动 / 停止的 Celery worker
# --------------------------------------------------------------------------- #
#: ``celery -A`` 要加载的应用
APP_PATH = "plugins.celery.celery_client"
#: 传递日志文件后缀的环境变量
LOG_SUFFIX_ENV = "MAS_LOG_SUFFIX"
#: worker 进程的日志文件后缀
WORKER_LOG_SUFFIX = ".worker"
#: 子进程状态轮询间隔（秒）
MONITOR_INTERVAL = 5.0
#: 停止时等待子进程退出的时间（秒），超时强杀
STOP_TIMEOUT = 10.0


def worker_command() -> list[str]:
    """拼装 worker 启动命令。

    用 ``sys.executable -m celery`` 而非 ``celery`` 可执行文件，保证子进程与
    当前进程使用同一解释器、同一套依赖。

    Returns:
        命令行参数列表。
    """
    command = [sys.executable, "-m", "celery", "-A", APP_PATH, "worker"]
    if os.name == "nt":
        command.append("--pool=solo")
    return command


def worker_env() -> dict[str, str]:
    """子进程环境变量：继承当前环境，并让日志写入独立文件。"""
    return {**os.environ, LOG_SUFFIX_ENV: WORKER_LOG_SUFFIX}


class CeleryWorker(Plugin):
    """Celery worker 子进程的生命周期管理。

    ``register_plugins()`` 时拉起 worker 子进程，``close_plugins()`` 时终止它。
    """

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._closing = False

    async def register(self) -> None:
        """拉起 worker 子进程，并挂一个观察任务跟踪其退出。

        启动失败（如依赖缺失、Celery 配置错误）由 :meth:`_watch` 以 ERROR
        记录退出码，不阻断 API 启动——与项目其它插件一致的 fail-open 策略。
        """
        if self.process is not None:
            return  # 幂等：重复注册不会拉起第二个 worker

        self._closing = False
        command = worker_command()
        logger.info("启动 Celery worker: {}", " ".join(command))
        # cwd 固定为项目根：worker 需按包路径导入 config / plugins
        worker_process = subprocess.Popen(command, cwd=config.BASE_DIR, env=worker_env())
        self.process = worker_process
        self._watch_task = asyncio.create_task(self._watch())
        logger.info(
            "Celery worker 已拉起 pid={} broker={}",
            worker_process.pid,
            config.CELERY_BROKER_URL or f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}",
        )

    async def _watch(self) -> None:
        """轮询子进程状态，退出即记录告警（不自动重启，便于暴露配置问题）。"""
        process = self.process
        if process is None:
            return
        while process.poll() is None:
            await asyncio.sleep(MONITOR_INTERVAL)
        if not self._closing:
            logger.error(
                "Celery worker 意外退出 pid={} code={}，异步任务将无人消费",
                process.pid,
                process.returncode,
            )

    async def close(self) -> None:
        """停止观察任务并终止 worker 子进程（超时则强杀）。"""
        self._closing = True
        if self._watch_task is not None:
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task
            self._watch_task = None

        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return

        logger.info("停止 Celery worker pid={}", process.pid)
        process.terminate()
        try:
            # 等待任务收尾：Celery 收到终止信号后做完当前任务再退出
            await asyncio.to_thread(process.wait, STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.warning("Celery worker 未在 {}s 内退出，强制结束", STOP_TIMEOUT)
            process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                await asyncio.to_thread(process.wait, STOP_TIMEOUT)
        logger.info("Celery worker 已停止")
