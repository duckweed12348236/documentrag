"""评估流水线的图定义：把生成与评测两个节点串成一条可执行的 LangGraph。

节点顺序::

    generate_dataset → eval_dataset

1. :func:`~plugins.agents.evaluation.nodes.generate_dataset`：``State.dataset_id`` 非空时
   **只做路由**、直接跳到评测节点（不读库、不重新生成）；否则用 ``State.content``（由
   :meth:`Evaluator.invoke` 读盘切分好的正文片段，节点内组装成文档对象）生成测试用例并落库
   为新数据集，写回 ``dataset_id`` 后同样跳到评测节点；
2. :func:`~plugins.agents.evaluation.nodes.eval_dataset`：按 ``dataset_id`` 从数据库读出用例，
   逐条执行检索问答、跑 Ragas 指标、让大模型解读评测结果，并把报告落库（同一数据集重新评测
   会覆盖旧报告）；随后**跳回生成节点**，开始下一轮。

节点之间的跳转**全部由各节点自己返回的 ``Command(goto=...)`` 决定**（与
:mod:`plugins.agents.ingestion.graph` 一致）：生成节点把流程交给评测节点，评测节点成功与
失败都跳回生成节点——``dataset_id`` 非空时它只做路由、立刻跳回评测节点，于是图**永远循环**；
只有输入不可用（``document_path`` 与 ``dataset_id`` 同时为空、数据集不存在、没有可评测用例）
时才跳 ``END``。因此本模块只负责按名字注册节点并声明入口（``START`` → :data:`NODES` 中的
首个节点），不设置任何节点之间的静态边；:data:`NODES` 的元组顺序只决定入口，不再决定流转
顺序。各节点的步骤进度通过 ``runtime.stream_writer`` 以
:class:`~plugins.agents.evaluation.schemas.Step` 上报，:meth:`Evaluator.invoke` 以流式方式
执行、逐个产出。

循环怎么停下来：:class:`Evaluator` 是进程内单例插件，``register`` 时建 PostgreSQL 连接池、
装上 :class:`~plugins.agents.checkpointers.CachedAsyncPostgresSaver`（PostgreSQL 持久化 +
Redis 缓存）并编译**带记忆**的图，同时在评测节点之前设断点（:data:`INTERRUPT_BEFORE`）。
评测节点跳回生成节点、又走到评测节点前时被断点拦下，本次 :meth:`Evaluator.invoke` 正好返回
——**一次调用一轮**。调用方不需要知道断点：再调一次 ``invoke`` 就是续跑下一轮（``invoke``
自己看会话是否停在断点），次数不限；每轮用例都重新从库里读、每条报告都覆盖旧报告，
``dataset_id`` 已写进会话状态，生成节点只做路由、不会重复生成数据集。会话标识（雪花 ID，
checkpointer 里按字符串存取）落在数据集上（:attr:`~models.dataset.Dataset.thread_id`），即
「一个数据集 ↔ 一条记忆线程」；不再需要这份记忆时用 :meth:`Evaluator.forget` 清除
（PostgreSQL checkpoint 与 Redis 缓存一并删除）。

checkpointer 的序列化器按
:data:`~plugins.agents.evaluation.schemas.CHECKPOINT_TYPES` 显式放行流水线状态里的自定义
类型（用例、报告等），否则 langgraph 反序列化时会告警、未来版本还会把未登记类型拦成原始
字典。

运行时上下文（大模型、向量模型、测试用例生成器、检索智能体与评测指标）由
:meth:`Evaluator.invoke` 自己构造，调用方只需给出输入。

用法::

    from plugins.agents.evaluation.graph import Evaluator
    from plugins.snowflake import Snowflake

    evaluator = Evaluator()                                              # 进程内单例
    await evaluator.register()                                           # 插件机制在启动时调用
    thread_id = Snowflake.get_id()                                       # 会话标识：雪花 ID
    async for step in evaluator.invoke(document_path=markdown_path, size=10, thread_id=thread_id):
        print(step.message, step.status)                                 # 生成数据集，停在评测节点前
    async for step in evaluator.invoke(dataset_id=dataset_id, thread_id=thread_id):  # 每调一次评测一轮
        print(step.message, step.status)
    await evaluator.forget(thread_id)                                    # 清除这条记忆（含 Redis 缓存）
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from openai import AsyncOpenAI
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import LangchainLLMWrapper, llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    ContextRelevance,
    Faithfulness,
)
from ragas.testset import TestsetGenerator

import config
from logger import logger
from plugins import Plugin
from plugins.agents.checkpointers import CachedAsyncPostgresSaver
from plugins.agents.evaluation.nodes import (
    EVAL_DATASET_NODE,
    GENERATE_DATASET_NODE,
    eval_dataset,
    generate_dataset,
)
from plugins.agents.evaluation.schemas import (
    CHECKPOINT_TYPES,
    Context,
    State,
    Step,
    StepStatus,
)
from plugins.agents.models import ds, qwen_embedding
from plugins.agents.retrieval.graph import Retriever
from plugins.md_utils import split, split_by_header
from plugins.snowflake import Snowflake

#: 节点函数签名：接收流水线状态与运行时（携带 :class:`~plugins.agents.evaluation.schemas.Context`），
#: 返回状态更新指令
NodeFunc = Callable[[State, Runtime[Context]], Awaitable[Command]]

#: 流水线节点：``(节点名, 节点函数)``，首项为入口；节点之间的流转由各自的
#: ``Command(goto=...)`` 决定，元组顺序不再决定执行顺序
NODES: Sequence[tuple[str, NodeFunc]] = (
    (GENERATE_DATASET_NODE, generate_dataset),
    (EVAL_DATASET_NODE, eval_dataset),
)

__all__ = ["INTERRUPT_BEFORE", "NODES", "RAGAS_MAX_TOKENS", "Evaluator", "NodeFunc"]

#: 断点位置：``eval_dataset`` 之前。评测节点结束会跳回生成节点、又走回这里，断点把它拦下，
#: 因此一次 :meth:`Evaluator.invoke` 只跑一轮就返回。断点要靠 checkpoint 才能恢复，所以只在
#: 带 checkpointer（插件已注册）的图上生效。
INTERRUPT_BEFORE: tuple[str, ...] = (EVAL_DATASET_NODE,)

#: Ragas instructor 调用的最大输出 token 数。Ragas 默认只给 1024，指标（尤其 Faithfulness
#: 的语句拆分）在长回答 / 多上下文时会被截断并抛 ``IncompleteOutputException``；DeepSeek
#: 单次输出上限是 8192，这里直接取满。
RAGAS_MAX_TOKENS = 8192


class _RagasEmbeddings(BaseRagasEmbedding):
    """把 LangChain 的 :class:`~langchain_core.embeddings.Embeddings` 适配成 Ragas 现代
    向量接口（``BaseRagasEmbedding``）。

    Ragas 0.4.x 的现代指标（如 ``AnswerRelevancy``）只接受 ``BaseRagasEmbedding``；这里把
    项目的文本向量模型（``qwen_embedding``）包一层复用同一套实现。
    """

    def __init__(self, embeddings: Embeddings) -> None:
        super().__init__()
        self._embeddings: Embeddings = embeddings

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        """同步单条向量化。"""
        vectors = self._embeddings.embed_documents([text])
        return vectors[0] if vectors else []

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        """异步单条向量化。"""
        vectors = await self._embeddings.aembed_documents([text])
        return vectors[0] if vectors else []


class Evaluator(Plugin):
    """评估智能体：进程内单例插件，对一份文档或一个已有数据集跑一轮评估流水线。

    ``__init__`` 先按**无记忆**编译一次（进程未注册时也能直接用），``register`` 时再编译
    一份带短期记忆的图，``close`` 时释放连接池与缓存客户端。图不设断点：一次
    :meth:`invoke` 从头跑到 ``END`` 就是一轮，想再评测就带着同一条 ``thread_id`` 再调一次。
    """

    def __init__(self) -> None:
        self._pool: AsyncConnectionPool | None = None
        self._checkpointer: CachedAsyncPostgresSaver | None = None
        #: 执行用的图：``register`` 时换成带短期记忆的版本
        self.graph: CompiledStateGraph = self._compile()

    def _compile(self, checkpointer: CachedAsyncPostgresSaver | None = None) -> CompiledStateGraph:
        """编译流水线；传入 checkpointer 即带上短期记忆与评测节点前的断点。"""
        builder = StateGraph(State, context_schema=Context)
        for name, node in NODES:
            builder.add_node(name, node)

        # 只声明入口；节点之间怎么走由各节点返回的 Command(goto=...) 决定
        builder.add_edge(START, NODES[0][0])

        # 断点要靠 checkpoint 才能恢复：没有 checkpointer 时不设断点，一次跑完（会一直循环
        # 到 recursion_limit，仅用于本地不注册插件时的调试）
        return builder.compile(
            checkpointer=checkpointer,
            interrupt_before=list(INTERRUPT_BEFORE) if checkpointer is not None else [],
        )

    @staticmethod
    def _build_config(thread_id: int) -> RunnableConfig:
        """按会话标识拼出运行配置（短期记忆按 ``thread_id`` 存取）。

        checkpointer 的 ``thread_id`` 是字符串列，雪花 ID 在这里转成字符串再交给它。
        """
        return {"configurable": {"thread_id": str(thread_id)}}

    async def invoke(
            self,
            size: int | None = None,
            document_path: str | Path | None = None,
            dataset_id: int | None = None,
            thread_id: int | None = None,
    ) -> AsyncGenerator[Step, None]:
        """流式执行一轮评估流水线，逐个产出步骤进度。

        用参数区分走哪条路，一次调用正好一轮：

        - 给了 ``dataset_id``：数据集已存在，会话停在评测节点前——不传输入、直接续跑一轮，
          评测完跳回生成节点、又被断点拦下，于是本次调用返回；
        - 没给 ``dataset_id``：给了 ``document_path`` 就读盘切分、生成测试用例并落库，交给
          评测节点，断点在评测节点前把它拦下。

        所以同一会话可以无限反复调用本方法评测：``dataset_id`` 已写进会话状态，生成节点只
        做路由，不会重复生成数据集；每轮用例都重新从库里读、每条报告都覆盖旧报告。文档在
        这里读、在这里切：按标题切章节、按 token 切小块（与入库流水线一致），只保留正文。

        Args:
            document_path: 本地 Markdown 文档路径；只在 ``dataset_id`` 为空时使用。
            dataset_id: 数据集主键；非空时续跑一轮评测（并覆盖旧报告）。
            size: 生成的测试用例条数（走 ``document_path`` 路径时使用）。
            thread_id: 短期记忆的会话标识（雪花 ID）；缺省时生成一个，并随数据集落库
                （见 :attr:`~models.dataset.Dataset.thread_id`）。

        Returns:
            异步生成器：各节点通过 ``runtime.stream_writer`` 上报的
            :class:`~plugins.agents.evaluation.schemas.Step`，步骤开始为 ``running``、
            结束为 ``success``、跳过与失败为 ``error``（带原因）。文档读不出来时产出
            「正在读取文档」的失败步骤后结束（不进入图执行）。
        """
        if not thread_id:
            thread_id = Snowflake.get_id()
        config = self._build_config(thread_id)

        # 文档在这里读、在这里切：结果随状态交给生成节点（也会随 checkpoint 一起保存）
        content = []
        if dataset_id is None and document_path:
            yield Step(message="正在读取文档")
            try:
                path = Path(document_path)
                text = await asyncio.to_thread(path.read_text, encoding="utf-8")
                content = [
                    piece
                    for _title, _line_no, body in split_by_header(
                        text.replace("\r\n", "\n").replace("\r", "\n")
                    )
                    for piece in split(body)
                ]
            except OSError as exc:  # 含 FileNotFoundError
                logger.opt(exception=exc).error(
                    "评估流水线文档读取失败 path={}", document_path
                )
                yield Step(message="正在读取文档", status=StepStatus.ERROR, error=str(exc))
                return
            yield Step(message="正在读取文档", status=StepStatus.SUCCESS)

        # Ragas instructor 的 LLM：max_tokens 必须显式放大，默认的 1024 会截断指标输出
        # （如 Faithfulness 的语句拆分）并抛 IncompleteOutputException
        client = AsyncOpenAI(
            api_key=ds.openai_api_key.get_secret_value(),
            base_url=str(ds.openai_api_base),
        )
        model = llm_factory(
            ds.model_name,
            provider="openai",
            client=client,
            max_tokens=RAGAS_MAX_TOKENS,
        )
        context = Context(
            model=ds,
            generator=TestsetGenerator(
                llm=LangchainLLMWrapper(ds),
                embedding_model=LangchainEmbeddingsWrapper(qwen_embedding),
            ),
            retriever=Retriever(),
            metrics=[
                ContextRelevance(model),
                ContextRecall(model),
                ContextPrecisionWithReference(model),
                Faithfulness(model),
                AnswerRelevancy(model, _RagasEmbeddings(qwen_embedding)),
            ],
        )
        # 评测路径（dataset_id 非空）不传输入：会话停在评测节点前，接着跑；
        # 生成路径带上输入，从 START 开始
        state = None if dataset_id is not None else State(size=size, content=content)
        logger.info(
            "评估流水线开始 document={} documents={} dataset={} thread={}",
            document_path,
            len(content),
            dataset_id,
            thread_id,
        )
        async for step in self.graph.astream(
                state,
                config=config,
                context=context,
                stream_mode="custom",
        ):
            yield step
        logger.info("评估流水线结束（评测节点前暂停或已结束） thread={}", thread_id)

    async def forget(self, thread_id: int) -> None:
        """清除某个会话的短期记忆（PostgreSQL checkpoint 与 Redis 缓存一并删除）。

        只删记忆，不动数据集及其用例 / 报告；会话不存在时无操作。

        Args:
            thread_id: 要清除的会话标识（雪花 ID，数据集上的 ``thread_id``）。
        """
        if self._checkpointer is None:
            logger.warning("评估记忆未初始化（未注册），跳过清除 thread={}", thread_id)
            return
        # adelete_thread 同时清两处存储：PostgreSQL 的 checkpoint 与 Redis 的缓存记录
        # （缓存键为 stm:memory:{thread_id}:*），只清前者会让缓存在 TTL 内继续被读到；
        # checkpointer 的会话标识是字符串，这里统一转成字符串
        await self._checkpointer.adelete_thread(str(thread_id))
        logger.info("评估会话的短期记忆已清除（含 Redis 缓存） thread={}", thread_id)

    async def register(self) -> None:
        """初始化 PostgreSQL 连接池与短期记忆，并编译带记忆（无断点）的评估图。

        checkpointer 注入的序列化器把 :data:`~plugins.agents.evaluation.schemas.CHECKPOINT_TYPES`
        加入 msgpack 白名单——状态里的自定义类型（用例、报告等）据此可原样还原；其余类型
        沿用 langgraph 的内置安全集合（消息、Command 等），未登记的自定义类型会被拦截成
        原始字典。
        """
        conninfo = (
            f"host={config.POSTGRES_HOST} "
            f"port={config.POSTGRES_PORT} "
            f"dbname={config.POSTGRES_DATABASE} "
            f"user={config.POSTGRES_USER} "
            f"password={config.POSTGRES_PASSWORD}"
        )
        pool = AsyncConnectionPool(
            conninfo=conninfo,
            min_size=config.POSTGRES_POOL_MIN_SIZE,
            max_size=config.POSTGRES_POOL_MAX_SIZE,
            open=False,  # 延迟到事件循环中显式开启
            kwargs={
                # checkpoint 依赖的必备连接参数
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )
        await pool.open(wait=True)
        self._pool = pool

        checkpointer = CachedAsyncPostgresSaver(
            pool,
            # 状态里的用例 / 报告等自定义类型显式登记进 msgpack 白名单（见 CHECKPOINT_TYPES）：
            # 默认序列化器只对未登记类型告警、未来版本会直接拦截，届时还原出来的只是原始
            # 字典，报表就无法再按对象取值
            serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES),
        )
        # 幂等建表：首次运行创建 checkpoint 相关表，之后仅校验迁移版本
        await checkpointer.setup()
        self._checkpointer = checkpointer

        self.graph = self._compile(checkpointer)
        logger.info(
            "评估流水线已就绪：短期记忆已启用（PostgreSQL checkpoint 表就绪，Redis 缓存 TTL {}s），"
            "不设断点 nodes={}",
            checkpointer.ttl, [name for name, _ in NODES],
        )

    async def close(self) -> None:
        """关闭短期记忆与 PostgreSQL 连接池，释放连接。"""
        if self._checkpointer is not None:
            await self._checkpointer.aclose()
            self._checkpointer = None
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        logger.info("评估流水线已关闭：短期记忆与 PostgreSQL 连接池已释放")
