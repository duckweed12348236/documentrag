from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from tortoise import Tortoise

import config
from logger import logger, setup_logger
from plugins import close_plugins, register_plugins

# 导入以触发 CeleryWorker 插件注册：PluginMeta 只登记「已导入」的插件类，
# 不导入本模块则 register_plugins() 不会拉起 worker 子进程。
from plugins.celery import CeleryWorker  # noqa: F401
from plugins.redis import close, connect, migrate
from controllers import document, evaluation, ingestion, retrieval, task


@asynccontextmanager
async def register(app: FastAPI):
    """应用生命周期：启动时初始化配置、日志、ORM 与插件，关闭时反向释放。"""
    setup_logger()
    logger.info("应用启动，日志目录: {}", config.LOG_DIR)
    # 1. ORM 连接池懒打开（首次查询时建立）
    # _enable_global_fallback=True 让 FastAPI 请求处理（与 lifespan 不同 task）
    # 也能拿到同一份 TortoiseContext。
    await Tortoise.init(config=config.TORTOISE_ORM, _enable_global_fallback=True)
    logger.info(
        "Tortoise ORM 已初始化：default={}@{}/{} pool={}-{}",
        config.POSTGRES_USER,
        config.POSTGRES_HOST,
        config.POSTGRES_DATABASE,
        config.POSTGRES_POOL_MIN_SIZE,
        config.POSTGRES_POOL_MAX_SIZE,
    )

    # 2. 建表：safe=True 只创建缺失的表，已存在的表不做改动
    await Tortoise.generate_schemas(safe=True)
    logger.info("业务表已就绪：{}", ", ".join(config.TORTOISE_MODELS))

    # 3. Redis：先建进程常驻连接（任务记录的提交 / 轮询都复用它，worker 子进程
    #    各有一条自己的），再做模型索引迁移（迁移复用这条连接）。两者都 fail-open：
    #    Redis 不可用只告警，不影响启动与模型的主键读写——提交任务的接口会以 503
    #    明确拒绝。
    try:
        await connect()
        logger.info("任务记录连接已就绪")
    except Exception as exc:  # noqa: BLE001 - Redis 不可用不应阻断启动
        logger.warning("任务记录连接建立失败（提交任务将返回 503）: {}", exc)
    await migrate()

    # 4. 注册所有插件：CeleryWorker 在此拉起 Celery worker 子进程（无需手动
    #    执行 celery worker），Agent 初始化智能体与短期记忆。
    #    注册放在 try 内：任一插件启动失败时，finally 仍会回收已启动的资源，
    #    避免 worker 子进程变成孤儿进程。
    try:
        await register_plugins()
        yield
    finally:
        # 5. 关闭时按注册顺序释放：插件（含 worker 子进程）→ Redis 常驻连接
        #    → ORM 连接池
        await close_plugins()
        await close()
        await Tortoise.close_connections()
        logger.info("应用关闭")
        await logger.complete()  # 等待异步队列中剩余日志写入完成


app = FastAPI(lifespan=register)
app.include_router(ingestion.router)
app.include_router(retrieval.router)
app.include_router(evaluation.router)
app.include_router(task.router)
app.include_router(document.router)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        port=8000,
        loop="asyncio:SelectorEventLoop",
    )
