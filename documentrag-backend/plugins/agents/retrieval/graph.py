"""产品检索流水线的图定义：把各节点串成一条可执行的 LangGraph。

节点顺序::

    understand_intention → rewrite_question → generate_hypothesis
        → search_online → recall → rerank → generate_answer

1. :func:`~plugins.agents.retrieval.nodes.understand_intention`：判定最新问题是否属于产品咨询；
2. :func:`~plugins.agents.retrieval.nodes.rewrite_question`：把问题改写成不依赖上下文即可读懂的独立问题；
3. :func:`~plugins.agents.retrieval.nodes.generate_hypothesis`：生成假设性回答，供向量召回；
4. :func:`~plugins.agents.retrieval.nodes.search_online`：联网搜索（可选环节，见下）；
5. :func:`~plugins.agents.retrieval.nodes.recall`：问题 + 假设性回答两路向量检索后 RRF 融合；
6. :func:`~plugins.agents.retrieval.nodes.rerank`：本地与网络候选一并重排、截断、融合；
7. :func:`~plugins.agents.retrieval.nodes.generate_answer`：依据片段作答并标注引用。

跳转：与入库 / 评测流水线不同，本流水线的节点**只在终止时**返回 ``Command(goto=END)``
（判定不通过、询问对象不明确需要反问、调用或解析失败），成功时返回 ``Command(update=...)``
交给图上的边决定下一节点——因此节点顺序在这里声明：``START`` → 首个节点，其后逐节点连边，
末端节点自然结束。

短期记忆（会话状态）：:class:`Retriever` 使用
:class:`~plugins.agents.checkpointers.CachedAsyncPostgresSaver`（PostgreSQL 持久化 + Redis
缓存）作为 checkpointer——``register`` 时建连接池、建表并编译**带记忆**的图，``close`` 时
释放。带记忆的图按会话调用（``configurable.thread_id``）：同一 ``thread_id`` 的状态（问题、
改写结果、检索片段、答案，以及各节点追加进 ``messages`` 的结论）会保存与恢复，调用方只需
给出本轮的 ``question``。checkpointer 的序列化器按
:data:`~plugins.agents.retrieval.schemas.CHECKPOINT_TYPES` 显式放行流水线状态里的自定义类型
（chunk 等），否则 langgraph 反序列化时会告警、未来版本还会把未登记类型拦成原始字典。

评测（:meth:`Retriever.eval`）走另一张**不带 checkpointer** 的图：批量评测是单轮、无状态的，
既不该要求 ``thread_id``，也不该往记忆库里写每条用例的临时会话。它以流式方式跑完整张图、
**收集**各节点上报的步骤，再从步骤里取最终答案与检索到的块，连同传入的用例标识一并返回。

模型与外部客户端统一由 :class:`Retriever` 在构造时建好、以
:class:`~plugins.agents.retrieval.schemas.Context` 注入图运行时（``runtime.context``），
节点不在模块级共享模型实例：

- 对话大模型（:data:`~plugins.agents.models.ds`）→ ``Context.model``，节点一律通过
  :func:`~plugins.agents.models.invoke` 调用；
- 文本向量模型（:data:`~plugins.agents.models.qwen_embedding`，按
  ``[tool.model.qwen-embedding]`` 构造）→ ``Context.embedding_model``，节点调用时把它传给
  :meth:`~plugins.agents.embedding.Embedding.embed`；
- 重排模型连接信息（按 ``[tool.model.qwen-rerank]`` 构造的
  :class:`~plugins.agents.models.RerankModel`）→ ``Context.rerank``，``rerank`` 节点把它
  交给重排客户端使用；
- 联网搜索客户端（按 ``[tool.tavily].key`` 即 :data:`config.TAVILY_KEY` 构造的
  :class:`~tavily.AsyncTavilyClient`）→ ``Context.tavily_client``，``search_online``
  节点直接用它检索。

用法::

    from plugins.agents.retrieval.graph import Retriever

    retriever = Retriever()                                          # 进程内单例
    async for step in retriever.invoke("iPhone 15 的保修期多久？", thread_id="session-1"):
        print(step.message, step.status)
    answer, chunks, key = await retriever.eval("iPhone 15 的保修期多久？", 0)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from tavily import AsyncTavilyClient

import config
from logger import logger
from plugins import Plugin
from plugins.agents.checkpointers import CachedAsyncPostgresSaver
from plugins.agents.models import RerankModel, ds, qwen_embedding
from plugins.agents.retrieval.nodes import (
    generate_answer,
    generate_hypothesis,
    recall,
    rerank,
    rewrite_question,
    search_online,
    understand_intention,
)
from plugins.agents.retrieval.schemas import CHECKPOINT_TYPES, Chunk, Context, State, Step

#: 节点函数签名：接收流水线状态与运行时（携带
#: :class:`~plugins.agents.retrieval.schemas.Context`），返回状态更新指令
NodeFunc = Callable[[State, Runtime[Context]], Awaitable[Command]]

#: 流水线节点：``(节点名, 节点函数)``，元组顺序即执行顺序（首项为入口）
NODES: Sequence[tuple[str, NodeFunc]] = (
    ("understand_intention", understand_intention),
    ("rewrite_question", rewrite_question),
    ("generate_hypothesis", generate_hypothesis),
    ("search_online", search_online),
    ("recall", recall),
    ("rerank", rerank),
    ("generate_answer", generate_answer),
)

__all__ = ["NODES", "Retriever", "NodeFunc"]


class Retriever(Plugin):
    """产品检索智能体：进程内单例，对一个问题跑完整条检索流水线。

    节点之间由本模块声明的静态边串联，节点自身只在失败时命令跳 ``END``。图在构造时先按
    **无记忆**编译一次（进程未注册时也能直接用），``register`` 时再编译出带短期记忆
    （:class:`~plugins.agents.checkpointers.CachedAsyncPostgresSaver`）的 ``graph``。
    对话大模型、重排模型连接信息与联网搜索客户端都在构造时建好，按运行时上下文注入。
    """

    def __init__(self) -> None:
        self._pool: AsyncConnectionPool | None = None
        self._checkpointer: CachedAsyncPostgresSaver | None = None
        #: 运行时上下文：模型与外部客户端都在这里注入，节点从 ``runtime.context`` 取用
        self._context: Context = Context(
            model=ds,
            embedding_model=qwen_embedding,
            rerank=RerankModel(
                url=config.RERANK_URL,
                name=config.RERANK_MODEL,
                key=config.RERANK_MODEL_KEY,
                instruct_field=config.RERANK_INSTRUCT_FIELD,
            ),
            tavily_client=AsyncTavilyClient(api_key=config.TAVILY_KEY),
        )
        #: 评测入口使用的图：始终无记忆——批量单轮，不要求 thread_id、也不往记忆库写会话
        self._stateless_graph: CompiledStateGraph = self._compile()
        #: 交互式入口使用的图：``register`` 时换成带短期记忆的版本
        self.graph: CompiledStateGraph = self._stateless_graph

    def _compile(self, checkpointer: CachedAsyncPostgresSaver | None = None) -> CompiledStateGraph:
        """编译流水线；传入 checkpointer 即带上短期记忆。"""
        builder = StateGraph(State, context_schema=Context)
        for name, node in NODES:
            builder.add_node(name, node)

        # 成功路径由图中的边决定：入口 → 逐节点串联，末端节点执行完即结束
        builder.add_edge(START, NODES[0][0])
        for (source, _), (target, _) in zip(NODES, NODES[1:]):
            builder.add_edge(source, target)

        return builder.compile(checkpointer=checkpointer)

    async def forget(self, thread_id: str) -> None:
        """清除某个会话的短期记忆（PostgreSQL checkpoint 与 Redis 缓存）。

        与 :meth:`~plugins.agents.evaluation.graph.Evaluator.forget` 不同，这里**不静默
        跳过、不吞异常**：调用方（删除会话）要在确认记忆清干净之后才删记录，否则记录没了、
        记忆还在，就再没有入口可以清理。

        Args:
            thread_id: 会话标识；该线程没有任何记忆时无操作。

        Raises:
            RuntimeError: 短期记忆未初始化（插件未注册）。
            Exception: PostgreSQL 或 Redis 清理失败（记忆可能残留）。
        """
        if self._checkpointer is None:
            raise RuntimeError(
                "检索会话的短期记忆未初始化（插件未注册），无法清除"
            )
        await self._checkpointer.adelete_thread(thread_id)
        logger.info("检索会话的短期记忆已清除 thread_id={}", thread_id)

    async def invoke(
            self,
            question: str,
            thread_id: str = "default",
    ) -> AsyncGenerator[Step, None]:
        """流式执行检索流水线，逐个产出步骤进度。

        本轮问题会作为一条消息追加进 ``state.messages``（后续节点的结论也逐条追加，配了短期
        记忆时按 ``thread_id`` 累积），各节点提交给模型的是「历史消息 + 最新问题」。同一
        ``thread_id`` 的状态由短期记忆保存与恢复。模型与联网搜索客户端都走构造时注入的
        运行时上下文。

        Args:
            question: 本轮用户问题。
            thread_id: 会话标识，相同的值共享同一份会话状态。

        Yields:
            各节点通过 ``runtime.stream_writer`` 上报的
            :class:`~plugins.agents.retrieval.schemas.Step`：节点开始为 ``running``、
            结束为 ``success`` / ``error``；答案生成节点的成功步骤里还会带上 ``answer``，
            便于随步骤流式下发。
        """
        context = self._context
        config_ = {"configurable": {"thread_id": thread_id}}
        state = State(question=question, messages=[HumanMessage(content=question)])
        logger.info("检索流水线开始 question={} thread={}", question, thread_id)
        async for step in self.graph.astream(
                state, config=config_, context=context, stream_mode="custom"
        ):
            yield step
        logger.info("检索流水线结束 question={} thread={}", question, thread_id)

    async def eval(self, question: str, key: int) -> tuple[str, list[Chunk], int]:
        """执行一条用例的检索问答，收集步骤后返回最终答案与检索块。

        走**不带短期记忆**的图：单轮、无状态，也不需要 ``thread_id``——评测衡量的是检索问答
        链路，且批量用例不应在记忆库里留下会话。模型与联网搜索客户端与交互式调用共用同一份
        运行时上下文。

        以流式方式跑完整张图，把各节点上报的
        :class:`~plugins.agents.retrieval.schemas.Step` **收集**起来，最后从步骤里取结果：
        重排节点完成时在步骤里写入检索到的块（``Step.chunks``），答案生成节点完成时在步骤里
        写入答案（``Step.answer``），其余步骤这两个字段均为空。

        Args:
            question: 用例问题。
            key: 数据集条目标识（调用方给出），原样回传，便于并发结果定位回该条用例。

        Returns:
            ``(answer, chunks, key)``：最终回答、检索到的块、传入的 ``key``。流水线中途失败
            （意图不通过、模型调用失败等）时答案为空串、块为空列表，由调用方按得分体现。
        """
        context = self._context
        state = State(question=question, messages=[HumanMessage(content=question)])
        logger.info("数据集用例检索开始 key={} question={}", key, question)

        # 自定义流逐个给出各节点的 Step；收集为列表后再取结果
        steps = [step async for step in self._stateless_graph.astream(
            state, context=context, stream_mode="custom"
        )]
        # 检索块与答案分别由重排、答案生成节点的完成步骤带上（其余步骤为空），取最近一次
        chunks = next(
            (step.chunks for step in reversed(steps) if step.chunks is not None), []
        )
        answer = next(
            (step.answer for step in reversed(steps) if step.answer is not None), ""
        )

        logger.info(
            "数据集用例检索结束 key={} steps={} answer_chars={} chunks={}",
            key, len(steps), len(answer), len(chunks),
        )
        return answer, chunks, key

    async def register(self) -> None:
        """初始化 PostgreSQL 连接池与短期记忆，并编译带记忆的检索图。

        checkpointer 注入的序列化器把 :data:`~plugins.agents.retrieval.schemas.CHECKPOINT_TYPES`
        加入 msgpack 白名单——状态里的自定义类型（chunk 等）据此可原样还原；其余类型沿用
        langgraph 的内置安全集合（消息、Command 等），未登记的自定义类型会被拦截成原始字典。
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
            # 状态里的 chunk 等自定义类型显式登记进 msgpack 白名单（见 CHECKPOINT_TYPES）：
            # 默认序列化器只对未登记类型告警、未来版本会直接拦截，届时还原出来的只是原始
            # 字典，恢复会话时节点取 chunk 的属性就会失败
            serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES),
        )
        # 幂等建表：首次运行创建 checkpoint 相关表，之后仅校验迁移版本
        await checkpointer.setup()
        self._checkpointer = checkpointer

        self.graph = self._compile(checkpointer)
        logger.info(
            "检索流水线已就绪：短期记忆已启用（PostgreSQL checkpoint 表就绪，Redis 缓存 TTL {}s）nodes={}",
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
        logger.info("检索流水线已关闭：短期记忆与 PostgreSQL 连接池已释放")
