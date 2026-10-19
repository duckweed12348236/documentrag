"""Redis 基础设施：RedisOM 模型定义、进程常驻连接与索引迁移。

RedisOM 的连接挂在模型类的 ``Meta.database`` 上，由各自的"使用方"注入：

- :class:`Task`：由本模块的 :func:`connect` 在**进程启动时**建好常驻连接并注入
  （API 进程在 ``app.register``、worker 进程在 Celery 启动信号里各调用一次），
  进程退出前用 :func:`close` 释放；:func:`migrate` 复用同一条连接。任务的建 /
  读 / 列表 / 进度 / 收尾都是 :class:`Task` 自己的方法。
- :class:`ShortTermMemory`：由
  :class:`plugins.agents.checkpointers.CachedAsyncPostgresSaver` 持有并注入它自己
  的连接（随该组件关闭）。

用法::

    from plugins.redis import ShortTermMemory, Task, TaskStatus, close, connect, migrate

    await connect()  # 建进程常驻连接并注入 Task（进程启动时调用一次）
    task = await Task.create(task_id, task_type="document_ingest",
                             inputs={...}, message="已受理")        # 提交方：状态 waiting
    await submit(task)   # 调度方：有空闲就立即执行，否则排队（见 plugins.celery.tasks）
    await task.add_step("正在解析文档")                                # 执行方
    await task.finish(status=TaskStatus.SUCCESS, outputs={...})        # 收尾
    tasks = await Task.list(status=TaskStatus.WAITING)                # 查询方：最新在前

    ShortTermMemory.Meta.database = redis_client  # 由 checkpointer 注入自己的连接
    record = await ShortTermMemory.get("thread_id:checkpoint_ns:latest")

    await migrate()  # 建立 RediSearch 索引，应用启动时执行一次即可
    ...
    await close()  # 进程退出前释放常驻连接

排队与并发：新建的任务状态为 :attr:`TaskStatus.WAITING`，由调度方（见
:mod:`plugins.celery.tasks`）取出全部排队任务、按创建时间（``Task.created_at``）取最早的
:data:`TASK_MAX_CONCURRENT` 个置为 ``running``；「同时执行多少个」直接由记录里
``status=running`` 的数量衡量，不另设计数器或执行槽，因此所有类型的任务合计同时至多
:data:`TASK_MAX_CONCURRENT` 个在执行，其余留在 ``waiting``。

Note:
    模型基于 RedisJSON（``JSON.SET`` / ``JSON.GET``），要求 Redis 已加载
    ReJSON 模块（Redis Stack / ``redis-stack-server`` 镜像）。裸 Redis 上
    缓存读写会失败，按事务方的 fail-open 策略降级为直连 PostgreSQL；
    :func:`migrate` 同样只告警、不阻断启动。
"""
from datetime import datetime, timezone
from enum import StrEnum
from functools import reduce
from operator import and_
from typing import Any

from aredis_om import Field, JsonModel, Migrator, NotFoundError
from redis.asyncio import Redis

import config
from logger import logger

__all__ = [
    "PROGRESS_KEY",
    "TASK_MAX_CONCURRENT",
    "ShortTermMemory",
    "Task",
    "TaskStatus",
    "build_redis_url",
    "close",
    "connect",
    "migrate",
]


class TaskStatus(StrEnum):
    """任务状态：``waiting`` 排队中 / ``running`` 执行中 / ``success`` 已完成 / ``error`` 已失败。"""

    WAITING = "waiting"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


#: 同一时刻允许执行的任务数上限（所有任务类型共用）
TASK_MAX_CONCURRENT = 5

#: :attr:`Task.outputs` 中存放进度步骤的键：进度与产出同处一个字段
PROGRESS_KEY = "progress"

#: 进程常驻的 Redis 连接：由 :func:`connect` 构造一次，:func:`close` 释放
_redis: Redis | None = None


def build_redis_url() -> str:
    """根据 :mod:`config` 中的 Redis 配置拼接连接串。"""
    auth = f":{config.REDIS_PASSWORD}@" if config.REDIS_PASSWORD else ""
    return f"redis://{auth}{config.REDIS_HOST}:{config.REDIS_PORT}/{config.REDIS_DB}"


class ShortTermMemory(JsonModel, index=True):
    """短期记忆缓存记录：一条记录承载一个 checkpoint 的完整状态。

    记录以 JSON 文档存放，``payload`` 是 langgraph ``serde`` 序列化后的负载
    （类型名见 ``serde_type``），内容与 PostgreSQL 中持久化的 checkpoint
    一致，可无损还原 ``CheckpointTuple``；RedisOM 会把 ``bytes`` 字段按
    base64 存取，二进制负载不会撞上 JSON 的编码限制。``checkpoint_id``
    单独成字段，便于用一次 ``JSON.GET <key> $.checkpoint_id`` 判断“最新”
    记录是否仍指向它。

    键名由 ``Meta`` 前缀与主键拼成 ``stm:memory:{pk}``。``index=True`` 是
    RedisOM 对 ``save()`` / ``key()`` 的硬性要求；读写都走主键、不做
    ``find()`` 检索，索引只由 :func:`migrate` 按需建立。

    Note:
        RedisOM 的元数据挂在类上（``Meta.database``），因此同一进程内应
        共用同一个 Redis 连接；重新注入会同时影响所有使用方。
    """

    class Meta:
        global_key_prefix = "langgraph"
        model_key_prefix = "memory"
        database = None  # 由使用方注入

    checkpoint_id: str
    serde_type: str
    payload: bytes


class Task(JsonModel, index=True):
    """后台任务记录：一条记录对应一次提交的异步任务（与具体业务无关）。

    任务以雪花 ID 作主键（提交时由
    :meth:`~plugins.snowflake.Snowflake.get_id` 生成），键名为
    ``task:record:{id}``；``type`` 标明任务类型（如 ``document_ingest``），
    ``status`` 与 ``type`` 都建了索引，可按状态或类型筛选。

    任务的输入与产出各占一个字段，内容由任务类型自行约定；进度写在产出里：

    - :attr:`inputs`：输入信息，如 ``{"filename": ..., "pdf_path": ...}``；
    - :attr:`outputs`：产出，仅任务成功时写入；其中 :data:`PROGRESS_KEY`（``progress``）
      键下是按执行顺序累积的进度步骤，每步形如
      ``{"message": ..., "status": ..., "error": ...}``，末条即最新进展。

    状态流转：:meth:`create` 建出 ``waiting`` 的任务 → 调度方按创建时间先后
    （:attr:`created_at`）把最早的任务 :meth:`start` 为 ``running`` → :meth:`add_step`
    累积进度 → :meth:`finish` 收尾为 ``success`` / ``error``。「同时执行多少个」直接数
    ``status=running`` 的记录，因此**所有类型的任务同时至多
    :data:`TASK_MAX_CONCURRENT` 个**在执行，其余留在 ``waiting`` 排队。

    任务记录的操作都是本类的方法（:meth:`create` / :meth:`fetch` / :meth:`list` /
    :meth:`earliest_waiting` / :meth:`start` / :meth:`add_step` / :meth:`finish` /
    :meth:`retry` / :meth:`remove`），不要在别处直接操作模型。``index=True`` 是 RedisOM 对 ``save()`` / ``key()`` 的硬性要求：
    按主键读写不依赖索引，:meth:`list` 走 RediSearch 检索，索引由 :func:`migrate` 建立。

    Note:
        连接由 :func:`connect` 在进程启动时注入 ``Meta.database``，本类方法直接
        使用它、不再自行连接；与 :class:`ShortTermMemory` 一样，同一进程内应共用
        同一个连接。
    """

    class Meta:
        global_key_prefix = "redis"
        model_key_prefix = "task"
        database = None  # 由使用方注入

    #: 主键：雪花算法生成的 64 位 ID，同时决定 Redis 键名
    id: int = Field(primary_key=True)
    #: 任务类型，由各任务自行约定，如 ``document_ingest``
    type: str = Field(index=True)
    #: 任务状态，见 :class:`TaskStatus`；新建即排队，由调度方 :meth:`start` 置为执行中
    status: TaskStatus = Field(default=TaskStatus.WAITING, index=True)
    #: 任务创建时间（UTC）：排队按它的先后执行，查询接口也据此展示受理时间
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    #: 失败原因（``status`` 为 ``error`` 时带上）
    error: str | None = None
    #: 任务输入：由任务类型自行约定，如 ``{"document_id": ..., "size": ...}``
    inputs: dict[str, Any] = Field(default_factory=dict)
    #: 任务产出：仅任务成功时写入业务产出，由任务类型自行约定；
    #: :data:`PROGRESS_KEY` 键下是按执行顺序累积的进度步骤（末条即最新进展），见类文档
    outputs: dict[str, Any] = Field(default_factory=dict)
    #: RedisOM 的默认主键字段：以 :attr:`id` 作主键后它不再参与键名计算，
    #: 仍会被默认值工厂填上一个无意义的 ULID，故置空并从存储中排除
    pk: str | None = Field(default=None, exclude=True)

    # ------------------------------------------------------------------ #
    # 任务记录操作
    # ------------------------------------------------------------------ #
    @classmethod
    async def create(
            cls, task_id: int, *, task_type: str, inputs: dict[str, Any], message: str
    ) -> "Task":
        """建任务记录并落库：状态 ``waiting``，写入输入信息与首条进度。

        Args:
            task_id: 任务 ID（雪花算法生成，同时决定 Redis 键名）。
            task_type: 任务类型。
            inputs: 任务的输入信息，存到 :attr:`inputs`。
            message: 首条进度的说明——记的是**已经完成**的一步（如「PDF 已保存」「任务已受理」）。

        Returns:
            写好的任务记录（状态 ``waiting``，等待调度方投递）。
        """
        task = cls(
            id=task_id,
            type=task_type,
            inputs=inputs,
            outputs={
                PROGRESS_KEY: [{
                    "message": message,
                    "status": TaskStatus.SUCCESS.value,
                    "error": None,
                }],
            },
        )
        await task.save()
        return task

    @classmethod
    async def fetch(cls, task_id: int) -> "Task | None":
        """按任务 ID 读取记录。

        Args:
            task_id: 任务 ID。

        Returns:
            任务记录；不存在（ID 错误）时返回 ``None``。
        """
        try:
            return await cls.get(task_id)
        except NotFoundError:
            return None

    @classmethod
    async def list(
            cls,
            status: TaskStatus | None = None,
            task_type: str | None = None,
            limit: int | None = None,
    ) -> list["Task"]:
        """列出任务记录，最新（主键最大）在前。

        走 RediSearch 检索（``find().all()``），因此依赖 :func:`migrate` 建立的
        索引。

        Args:
            status: 只返回该状态的任务；缺省返回全部状态。
            task_type: 只返回该类型的任务（如 ``dataset_create``）；缺省返回全部类型。
            limit: 最多返回多少条；缺省不限制。

        Returns:
            任务记录列表，最新在前。
        """
        expressions = []
        if status is not None:
            expressions.append(cls.status == status)
        if task_type is not None:
            expressions.append(cls.type == task_type)
        # 多个条件用 AND 组合（find() 的多参数是 OR，不能直接并列传入）
        query = cls.find(reduce(and_, expressions)) if expressions else cls.find()
        tasks = await query.all()
        tasks.sort(key=lambda task: task.id, reverse=True)
        if limit is not None:
            tasks = tasks[:limit]
        return tasks

    @classmethod
    async def pop_earliest_waiting(cls, limit: int) -> "list[Task]":
        """按创建时间先后取排队中的前 ``limit`` 个任务。

        走 RediSearch 取出全部排队任务后在进程内按 :attr:`created_at` 排序（索引不建
        排序字段，队列长度有限，排序开销可忽略）。主键是雪花 ID，顺序本也一致，这里以
        创建时间为准。

        Args:
            limit: 最多取多少个。

        Returns:
            排队任务列表，创建时间早的在前；没有排队任务时为空列表。
        """
        tasks = await cls.list(status=TaskStatus.WAITING)
        tasks.sort(key=lambda task: task.created_at)
        return tasks[:limit]

    # ------------------------------------------------------------------ #
    # 状态流转：排队 → 执行中 → 收尾
    # ------------------------------------------------------------------ #
    async def start(self) -> bool:
        """把排队中的任务置为 ``running``，返回是否由本次调用完成状态切换。

        调度方（见 :mod:`plugins.celery.tasks`）按创建时间取出最早的排队任务后逐个调用本
        方法。这里重新读一次记录、只在仍为 ``waiting`` 时改写状态：并发的调度方（API 进程
        与多个 worker）取到同一条任务时，后到的那个会发现状态已经变了、直接跳过，不会把
        同一个任务投递两次。

        Returns:
            ``True`` 表示状态已置为 ``running``（本次负责投递）；``False`` 表示记录已不存在
            或已被别的调度方取走。
        """
        latest = await type(self).fetch(self.id)
        if latest is None or latest.status is not TaskStatus.WAITING:
            return False
        self.status = TaskStatus.RUNNING
        await self.save()
        return True

    async def add_step(
            self,
            message: str,
            status: TaskStatus = TaskStatus.RUNNING,
            error: str | None = None,
    ) -> None:
        """把一步进度追加到 :attr:`outputs` 的 :data:`PROGRESS_KEY` 并落库（末条即最新进展）。

        Args:
            message: 步骤说明，如「正在解析文档」。
            status: 该步骤的状态，默认 ``running``（步骤开始）。
            error: 失败原因（``status`` 为 ``error`` 时带上）。
        """
        steps = self.outputs.setdefault(PROGRESS_KEY, [])
        steps.append({
            "message": message,
            "status": status.value,
            "error": error,
        })
        await self.save()

    async def finish(
            self,
            *,
            status: TaskStatus,
            error: str | None = None,
            outputs: dict[str, Any] | None = None,
    ) -> None:
        """收尾任务记录并落库：写入最终状态，成功时一并写入产出。

        Args:
            status: 最终状态，``success`` 或 ``error``。
            error: 失败原因（``status`` 为 ``error`` 时带上）。
            outputs: 产出信息（仅成功时传入），写入 :attr:`outputs`；进度
                （:data:`PROGRESS_KEY`）原样保留。

        Note:
            排队中的任务**也必须**经由 :meth:`finish` 结束（例如投递失败时）——任务收尾后
            执行中的记录数才会下降，排在后面的任务靠它腾出的位置才会被投递。
        """
        self.status = status
        self.error = error
        if outputs is not None:
            self.outputs = {**outputs, PROGRESS_KEY: self.outputs.get(PROGRESS_KEY, [])}
        await self.save()

    async def retry(self) -> None:
        """把失败的任务重新置为排队，等调度方重新投递。

        清掉失败原因与上次的业务产出，并在进度末尾记一条「任务重新排队」；进度历史保留，
        便于对照上次失败到哪一步。输入信息（:attr:`inputs`）原样保留，重试时任务体据此重新
        执行。

        Note:
            只有 ``error`` 状态的任务可以重试，调用方负责判断（接口层对其它状态返回 409）；
            这里不校验状态，避免把「能不能重试」的业务规则散落到两处。
        """
        self.status = TaskStatus.WAITING
        self.error = None
        self.outputs = {PROGRESS_KEY: self.outputs.get(PROGRESS_KEY, [])}
        self.outputs[PROGRESS_KEY].append({
            "message": "任务重新排队",
            "status": TaskStatus.WAITING.value,
            "error": None,
        })
        await self.save()

    async def remove(self) -> None:
        """删除任务记录（从 Redis 中移除，不可恢复）。

        Note:
            执行中的任务**不应**删除：worker 收尾时会 ``save()``，记录会被重新写回，调度方也
            会继续把它算作执行中的一个。能不能删由调用方判断（接口层对 ``running`` 返回
            409），这里不校验状态。
        """
        # RedisOM 的 delete 是类方法，按主键删；实例上没有可直接调用的删除方法
        await type(self).delete(self.id)


async def connect() -> Redis:
    """构造（或复用）进程常驻的 Redis 连接，并注入 :class:`Task`。

    幂等：首次调用时按 :mod:`config` 建连接、``PING`` 探测可用性，成功后常驻本
    进程并注入 ``Task.Meta.database``；之后的调用直接返回同一条连接。两个进程
    各调用一次——API 进程在 ``app.register``，worker 进程在 Celery 启动信号
    （:func:`~plugins.celery._register_worker_plugins`）里，都在即将使用它的那个
    事件循环上完成构造。

    探测失败时连接不缓存并抛出：调用方按 fail-open 只记日志，后续任务再次调用时
    会重新构造，不会留下“永远拿不到连接”的状态。

    Returns:
        常驻连接（开启 ``decode_responses``——RedisOM 的读写按该约定解析）。

    Raises:
        Exception: Redis 不可用（连不上、认证失败等）时原样抛出。
    """
    global _redis
    if _redis is not None:
        return _redis

    connection = Redis.from_url(build_redis_url(), decode_responses=True)
    try:
        await connection.ping()
    except Exception:
        await connection.aclose()
        raise
    _redis = connection
    Task.Meta.database = connection
    return connection


async def close() -> None:
    """关闭进程常驻连接（幂等）；进程退出前由使用方调用。"""
    global _redis
    connection, _redis = _redis, None
    if connection is not None:
        await connection.aclose()


async def migrate() -> None:
    """按模型定义建立/更新 RediSearch 索引。

    迁移是幂等的：RedisOM 把每个模型的索引 schema 哈希记在 Redis 中，只有
    索引缺失或 schema 变更时才重建（``DROP`` + ``CREATE``），因此可以放心地
    在每次启动时执行。

    迁移用进程常驻连接（:func:`connect`）——没有就先建好，跑完不关闭它：连接的
    生命周期归进程，不归这次迁移。Redis 不可用或未加载 RediSearch 模块时迁移会
    失败，这里捕获后只记录告警——模型读写都走主键、不依赖索引检索，迁移失败不应
    阻断应用启动。

    Note:
        Redis 只允许在 0 号库创建检索索引：``config.REDIS_DB`` 非 0 时迁移
        必然失败（同样只告警），此时模型仍可正常读写。
    """
    try:
        await Migrator(conn=await connect()).run()
        logger.info("Redis 模型迁移完成：索引已就绪")
    except Exception as exc:  # noqa: BLE001 - 迁移失败不应阻断应用启动
        logger.warning("Redis 模型迁移失败（仅影响索引检索，主键读写不受影响）: {}", exc)
