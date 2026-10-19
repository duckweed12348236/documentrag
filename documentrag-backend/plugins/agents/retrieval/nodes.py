"""产品检索流水线的节点实现。

:func:`understand_intention`（意图识别）：

1. 从运行时上下文取对话大模型（``runtime.context.model``），把 ``state.question`` 作为
   用户最新问题拼成消息列表提交它（提示词见
   :data:`~plugins.agents.retrieval.prompts.INTENTION_UNDERSTANDING_PROMPT`）；
2. 调用统一走 :func:`~plugins.agents.models.invoke`（请求绑定
   ``response_format={"type": "json_object"}``），输出由
   :class:`~plugins.agents.retrieval.schemas.Intention` 自行解析并校验（该模型同时是
   校验器），得到
   ``{"continued": bool, "object_explicit": bool, "clarification": str | None}``——
   ``object_explicit`` 只判定询问对象是否明确（最新问题显式提及或由对话历史唯一确定），
   不输出对象名称；调用异常或校验失败时由 invoke 带上错误信息重试；
3. 判定不通过（``continued`` 为 ``False``）或重试仍失败时跳转尾节点结束流水线；询问对象
   不明确（``object_explicit`` 为 ``False``）时同样结束，但把反问语句作为本轮回答
   （``state.answer`` 与步骤的 ``answer``）下发；只有判定通过且对象明确才继续后续节点。

:func:`rewrite_question`（问题改写）：

1. 同样提交最新问题（提示词见
   :data:`~plugins.agents.retrieval.prompts.QUESTION_REWRITING_PROMPT`），把问题中的
   代词、省略主语补全为具体商品名称，改写成不依赖上下文即可读懂的独立问题；
2. 模型输出为单行纯文本，改写结果落到 ``state.rewritten_question``（原问题仍留在
   ``state.question`` 中），后续节点按改写后的问题检索；调用失败或返回为空时跳转
   尾节点。

:func:`generate_hypothesis`（假设性回答生成）：

1. 以改写后的问题（``state.rewritten_question``，缺失时回退 ``state.question``）
   提交大模型（提示词见
   :data:`~plugins.agents.retrieval.prompts.HYPOTHESIS_GENERATION_PROMPT`），生成一段
   文档口吻的假设性答案：不要求与真实资料一致，但要求术语与表述贴近产品文档，便于用
   它的向量召回真实片段；
2. 结果写入 ``state.hypothesis``；调用失败或返回为空时跳转尾节点。

:func:`search_online`（联网搜索）：

1. 以改写后的问题（``state.rewritten_question``，缺失时回退 ``state.question``）调用
   运行时上下文注入的搜索函数 ``Context.search_online``（见
   :class:`~plugins.agents.retrieval.schemas.Context`）——通常是百炼「联网搜索」MCP
   的查询包装，服务地址为 ``<百炼根地址>/mcps/WebSearch/mcp``（Streamable HTTP，
   Header 为 ``Authorization: Bearer <DASHSCOPE_API_KEY>``）；
2. 返回结果规范成 :class:`~plugins.agents.retrieval.schemas.WebChunk` 列表写入
   ``state.web_chunks``；上下文未注入搜索函数或调用失败时跳转尾节点。

:func:`recall`（向量检索）：

1. 改写后的问题（缺失时回退 ``state.question``）与 ``state.hypothesis`` 分别向量化、
   分别提交 :meth:`~plugins.milvus.Milvus.find_chunks`（没有假设性回答时只检索问题），
   在全部 chunk 中检索；
2. 两路结果按 **加权 RRF（倒数排名融合）** 合并：``得分 = Σ 权重 / (RRF_K + 名次)``，
   名次为各结果列表中的名次（从 1 起计，:data:`RRF_K` 为平滑常数），问题路与假设性
   回答路的权重分别为 :data:`QUESTION_WEIGHT` 与 :data:`HYPOTHESIS_WEIGHT`；按融合
   得分从高到低经 :func:`_build_local_chunk` 转成本地的
   :class:`~plugins.agents.retrieval.schemas.LocalChunk`（得分落到 ``score``，状态里
   不复用 milvus 的检索结果类型）写入 ``state.local_chunks``。

:func:`rerank`（重排 + 融合）：

1. 候选为 ``state.local_chunks``（本地向量检索结果）与 ``state.web_chunks``（联网搜索
   结果）：正文按「本地在前、网络在后」一次性提交
   :class:`~plugins.agents.rerank.RerankClient`，连接信息取运行时上下文的
   ``Context.rerank``（见 :class:`~plugins.agents.models.RerankModel`），按改写后的问题
   （缺失时回退 ``state.question``）得到每条候选 0~1 的相关度；
2. 相关度高的在前，按**固定上下限 + 断崖式阈值**截断（数量上限
   :data:`RERANK_MAX_CHUNKS`、下限 :data:`RERANK_MIN_CHUNKS`；从上往下比较相邻两项，
   跌幅超过 :data:`RERANK_MAX_GAP` 或超过前一项分数的 :data:`RERANK_MAX_GAP_RATIO`
   即为断崖）；
3. 保留下来的候选统一映射成
   :class:`~plugins.agents.retrieval.schemas.Chunk`（本地块 ``source="local"``、网络
   片段 ``source="online"``），按重排后的顺序写入 ``state.chunks``。

:func:`generate_answer`（答案生成）：

1. 按 :data:`~plugins.agents.retrieval.prompts.ANSWER_GENERATION_PROMPT` 约定的 JSON
   结构一次性提交：``question``（改写后的问题，缺失时回退 ``state.question``）与
   ``chunks``（每项仅 ``title`` / ``content``）；
2. 模型只依据片段作答、句末用 ``[序号]`` 标注引用（序号即 ``chunks`` 下标 + 1，调用方
   可按 ``state.chunks`` 同序映射回 ``url`` / ``filename``）；答案写入 ``state.answer``，
   同时在步骤的 ``answer`` 字段里带上，便于随步骤流式下发给前端。

跳转：节点的流转由自身返回的 ``Command(goto=...)`` 决定——判定通过 / 改写完成 /
假设性回答生成完成 / 联网搜索完成 / 向量检索完成 / 重排完成 / 答案生成完成时返回
``Command``，后续节点由图中的边决定；判定不通过、调用失败或解析失败时跳 ``END``。

消息列表：调用方把本轮用户问题以 ``HumanMessage`` 追加进 ``state.messages``，每个节点再把
本轮结论以一条 ``AIMessage`` 追加进来（如「识别用户意图，用户意图为产品咨询」）；各节点提交
给模型的是「历史消息 + 最新问题」（见 :func:`_conversation`）。配了短期记忆时按 ``thread_id``
累积，供调用方追踪、也供后续轮次参考。
"""

import json
from collections.abc import Mapping

from langchain.messages import AnyMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command

import config
from logger import logger
from plugins.agents.models import TextOutputValidator, invoke
from plugins.agents.retrieval.prompts import (
    ANSWER_GENERATION_PROMPT,
    HYPOTHESIS_GENERATION_PROMPT,
    INTENTION_UNDERSTANDING_PROMPT,
    QUESTION_REWRITING_PROMPT,
)
from plugins.agents.retrieval.schemas import (
    Chunk,
    Context,
    Intention,
    LocalChunk,
    State,
    Step,
    StepStatus,
    WebChunk,
)
from plugins.agents.embedding import Embedding
from plugins.milvus import ChunkHit, Milvus
from plugins.agents.rerank import RerankClient

#: 问题 / 假设性回答每次检索取回的 chunk 数量上限
CHUNK_SEARCH_LIMIT = 10
#: RRF（倒数排名融合）平滑常数：融合得分 = Σ 权重 / (RRF_K + 名次)，名次从 1 起计
RRF_K = 60
#: 问题路在 RRF 融合中的权重
QUESTION_WEIGHT = 1.0
#: 假设性回答路在 RRF 融合中的权重
HYPOTHESIS_WEIGHT = 1.0
#: 重排后保留的 chunk 数量上限（固定上限）
RERANK_MAX_CHUNKS = 20
#: 重排后至少保留的 chunk 数量（固定下限）
RERANK_MIN_CHUNKS = 1
#: 断崖判定的绝对阈值：相邻两块分数差超过该值即视为断崖
RERANK_MAX_GAP = 0.5
#: 断崖判定的相对阈值：相邻两块分数差超过前一块分数的该比例即视为断崖
RERANK_MAX_GAP_RATIO = 0.25
#: 兜底反问：模型判定询问对象不明确、却没有给出反问语句时使用
CLARIFICATION_FALLBACK = "请问您想咨询的是哪款产品？"


async def understand_intention(state: State, runtime: Runtime[Context]) -> Command:
    """判断用户最新问题是否属于产品咨询、询问对象是否明确，无法确定对象时反问用户。

    最新问题（``state.question``）提交运行时上下文的对话大模型（``runtime.context.model``），
    并绑定 ``response_format={"type": "json_object"}``，模型按
    :class:`~plugins.agents.retrieval.schemas.Intention` 的 schema 输出
    ``{"continued": bool, "object_explicit": bool, "clarification": str | None}``：询问
    对象是否明确——最新问题显式提及、或能由对话历史唯一确定即为明确（不输出对象名称）；
    不明确时模型给出一句反问语句。调用与解析统一走 :func:`~plugins.agents.models.invoke`，
    输出由 :class:`~plugins.agents.retrieval.schemas.Intention` 自行校验成已解析的对象
    （该模型同时是校验器），调用异常或校验失败时由 invoke 带上错误信息重试。

    Args:
        state: 流水线状态，使用其中的 ``question``（用户最新问题）。
        runtime: LangGraph 运行时（携带 :class:`~plugins.agents.retrieval.schemas.Context`
            上下文，模型 / 重排连接信息 / 搜索客户端等依赖按需从 ``runtime.context`` 取），
            步骤进度由 ``stream_writer`` 上报。

    Returns:
        ``Command``：判定通过且询问对象明确时返回空 ``Command``（不修改状态，后续节点由
        图中的边决定）；判定不通过时跳转尾节点；询问对象不明确时把反问语句写进
        ``answer``（并随完成步骤的 ``answer`` 一并下发）后跳转尾节点；重试后仍失败时同样
        跳转尾节点并上报 ``error`` 步骤。
    """
    writer = runtime.stream_writer

    logger.info("意图识别开始 question={}", state.question)
    writer(Step(message="正在识别用户意图"))

    # json mode 结构化输出 + 校验重试：拿到已校验的 Intention 或错误文本
    intention, error = await invoke(
        runtime.context.model,
        [
            SystemMessage(content=INTENTION_UNDERSTANDING_PROMPT),
            *_build_conversation(state),
        ],
        Intention,
        response_format={"type": "json_object"},
    )
    if error:
        logger.error("意图识别失败 question={} err={}", state.question, error)
        writer(Step(message="正在识别用户意图", status=StepStatus.ERROR, error=error))
        return Command(goto=END)

    if not intention.continued:
        # 判定不通过：最新问题不属于产品咨询，流水线到此为止
        logger.info("意图识别不通过：最新问题不属于产品咨询 question={}", state.question)
        writer(Step(message="正在识别用户意图", status=StepStatus.SUCCESS))
        return Command(
            update={"messages": [AIMessage(content="识别用户意图，用户意图不属于产品咨询")]},
            goto=END,
        )

    if not intention.object_explicit:
        # 询问对象不明确：检索无从下手，把反问语句作为本轮回答结束流水线
        clarification = intention.clarification or CLARIFICATION_FALLBACK
        logger.info(
            "意图识别需要反问：询问对象不明确 question={} clarification={}",
            state.question, clarification,
        )
        writer(Step(
            message="正在识别用户意图",
            status=StepStatus.SUCCESS,
            answer=clarification,
        ))
        return Command(
            update={
                "answer": clarification,
                "messages": [AIMessage(
                    content=f"识别用户意图，询问对象不明确，反问用户「{clarification}」"
                )],
            },
            goto=END,
        )

    logger.info("意图识别通过 question={}", state.question)
    writer(Step(message="正在识别用户意图", status=StepStatus.SUCCESS))
    return Command(update={
        "messages": [AIMessage(content="识别用户意图，用户意图为产品咨询")]
    })


async def rewrite_question(state: State, runtime: Runtime[Context]) -> Command:
    """把问题改写为不依赖上下文即可读懂的独立问题。

    改写规则由 :data:`~plugins.agents.retrieval.prompts.QUESTION_REWRITING_PROMPT`
    约定：把代词、省略主语补全为具体商品名称，不新增信息、不臆造
    商品名（无商品指向时原样返回）。模型只输出单行纯文本，改写结果落到
    ``state.rewritten_question``（原问题保留在 ``state.question`` 中），后续节点按
    改写后的问题检索。

    模型取运行时上下文的 ``runtime.context.model``，调用走
    :func:`~plugins.agents.models.invoke`（输出用
    :class:`~plugins.agents.models.TextOutputValidator` 只校验非空），调用异常或返回为空
    时由 invoke 重试。

    Args:
        state: 流水线状态，使用其中的 ``question``（用户最新问题）。
        runtime: LangGraph 运行时（携带 :class:`~plugins.agents.retrieval.schemas.Context`
            上下文，模型 / 重排连接信息 / 搜索客户端等依赖按需从 ``runtime.context`` 取），
            步骤进度由 ``stream_writer`` 上报。

    Returns:
        ``Command``：改写成功时以 ``rewritten_question`` 字段回传改写后的问题；
        重试后仍失败时跳转尾节点并上报 ``error`` 步骤。
    """
    writer = runtime.stream_writer

    logger.info("问题改写开始 question={}", state.question)
    writer(Step(message="正在改写用户问题"))

    rewritten, error = await invoke(
        runtime.context.model,
        [
            SystemMessage(content=QUESTION_REWRITING_PROMPT),
            *_build_conversation(state),
        ],
        TextOutputValidator,
    )
    if error:
        logger.error("问题改写失败 question={} err={}", state.question, error)
        writer(Step(message="正在改写用户问题", status=StepStatus.ERROR, error=error))
        return Command(goto=END)

    rewritten = rewritten.strip()
    logger.info("问题改写完成 question={} rewritten={}", state.question, rewritten)
    writer(Step(message="正在改写用户问题", status=StepStatus.SUCCESS))
    return Command(update={
        "rewritten_question": rewritten,
        "messages": [AIMessage(content=f"改写用户问题，改写后的问题为「{rewritten}」")],
    })


def _build_conversation(state: State, question: str | None = None) -> list[AnyMessage]:
    """按「历史消息 + 最新问题」拼出提交给模型的消息列表。

    历史消息取自 ``state.messages``（含此前各轮的消息，可为空）；最新问题始终作为**最后
    一条**——即提示词约定的「对话中的最后一条」——``question`` 为空时用 ``state.question``
    （如传入改写后的问题，则消息以改写结果结尾）。

    当前问题已由调用方追加进 ``state.messages`` 且正好在末尾时不再重复追加。
    """
    messages = list(state.messages or [])
    latest = question or state.question
    if messages and isinstance(messages[-1], HumanMessage) and messages[-1].content == latest:
        return messages
    return [*messages, HumanMessage(content=latest)]


embedder = Embedding()
milvus = Milvus()
#: 重排客户端：进程内复用；它本身无状态，连接信息每次由运行时上下文的
#: ``Context.rerank``（:class:`~plugins.agents.models.RerankModel`）给出
rerank_client = RerankClient()


async def generate_hypothesis(state: State, runtime: Runtime[Context]) -> Command:
    """生成假设性答案，写入 ``state.hypothesis`` 供向量检索使用。

    以改写后的问题（``state.rewritten_question``，缺失时回退 ``state.question``）
    提交运行时上下文的对话大模型（``runtime.context.model``），按
    :data:`~plugins.agents.retrieval.prompts.HYPOTHESIS_GENERATION_PROMPT` 生成一段
    文档口吻的假设性答案：内容不要求与真实资料一致，但需与查询高度相关、术语贴近
    产品文档，用它的向量去召回真实文档片段。调用走
    :func:`~plugins.agents.models.invoke`（输出用
    :class:`~plugins.agents.models.TextOutputValidator` 只校验非空）。

    Args:
        state: 流水线状态，使用其中的 ``rewritten_question`` 与 ``question``。
        runtime: LangGraph 运行时（携带 :class:`~plugins.agents.retrieval.schemas.Context`
            上下文，模型 / 重排连接信息 / 搜索客户端等依赖按需从 ``runtime.context`` 取），
            步骤进度由 ``stream_writer`` 上报。

    Returns:
        ``Command``：生成成功时以 ``hypothesis`` 字段回传假设性答案；重试后仍失败
        时跳转尾节点并上报 ``error`` 步骤。
    """
    writer = runtime.stream_writer

    # 检索依据为改写后的问题；改写缺失（未执行或失败）时回退到原问题
    question = state.rewritten_question or state.question
    logger.info("假设性回答生成开始 question={}", question)
    writer(Step(message="正在生成假设性回答"))

    hypothesis, error = await invoke(
        runtime.context.model,
        [
            SystemMessage(content=HYPOTHESIS_GENERATION_PROMPT),
            *_build_conversation(state, question),
        ],
        TextOutputValidator,
    )
    if error:
        logger.error("假设性回答生成失败 question={} err={}", question, error)
        writer(Step(message="正在生成假设性回答", status=StepStatus.ERROR, error=error))
        return Command(goto=END)

    hypothesis = hypothesis.strip()
    logger.info(
        "假设性回答生成完成 question={} chars={} hypothesis={}",
        question, len(hypothesis), hypothesis,
    )
    writer(Step(message="正在生成假设性回答", status=StepStatus.SUCCESS))
    return Command(update={
        "hypothesis": hypothesis,
        "messages": [AIMessage(content=f"生成假设性回答，假设性回答为「{hypothesis}」")],
    })


async def recall(state: State, runtime: Runtime[Context]) -> Command:
    """向量检索：问题与假设性回答分别检索后按 RRF 融合，写入 ``state.local_chunks``。

    1. 改写后的问题（缺失时回退 ``state.question``）与 ``state.hypothesis`` 连同运行时
      上下文的向量模型（``Context.embedding_model``）各转成稠密 + 稀疏向量，分别提交
      :meth:`~plugins.milvus.Milvus.find_chunks`（没有假设性回答时只检索问题），
      在全部 chunk 中检索；
    2. 两路结果按 **加权 RRF（倒数排名融合）** 合并：``得分 = Σ 权重 / (RRF_K + 名次)``，
      名次为各结果列表中的名次（从 1 起计，:data:`RRF_K` 为平滑常数），问题路与假设性
      回答路的权重分别为 :data:`QUESTION_WEIGHT` 与 :data:`HYPOTHESIS_WEIGHT`，同一
      chunk 被两路命中时按各自权重累加得分；
    3. 按融合得分从高到低排序，经 :func:`_build_local_chunk` 转成本地的
      :class:`~plugins.agents.retrieval.schemas.LocalChunk`（得分落到 ``score``，状态里
      不复用 milvus 的检索结果类型）后写入 ``state.local_chunks``。

    Args:
        state: 流水线状态，使用其中的 ``rewritten_question``、``question`` 与
            ``hypothesis``。
        runtime: LangGraph 运行时（携带 :class:`~plugins.agents.retrieval.schemas.Context`
            上下文，模型 / 重排连接信息 / 搜索客户端等依赖按需从 ``runtime.context`` 取），
            步骤进度由 ``stream_writer`` 上报。

    Returns:
        ``Command``：成功时以 ``local_chunks`` 字段回传融合后的检索结果；向量化或检索
        失败时跳转尾节点并上报 ``error`` 步骤。
    """
    writer = runtime.stream_writer

    question = state.rewritten_question or state.question
    hypothesis = state.hypothesis
    logger.info(
        "向量检索开始 question={} hypothesis_chars={}",
        question, len(hypothesis or ""),
    )
    writer(Step(message="正在检索相关片段"))

    try:
        # 1. 问题与假设性回答分别向量化、分别检索
        texts = [question, hypothesis] if hypothesis else [question]
        dense, sparse = await embedder.embed(
            runtime.context.embedding_model, texts, "both"
        )
        if not dense or not sparse:
            raise RuntimeError("文本向量返回为空")
        question_hits = await milvus.find_chunks(
            [dense[0]], [sparse[0]],
            limit=CHUNK_SEARCH_LIMIT,
        )
        hypothesis_hits = (
            await milvus.find_chunks(
                [dense[1]], [sparse[1]],
                limit=CHUNK_SEARCH_LIMIT,
            )
            if hypothesis else []
        )
    except Exception as exc:
        logger.opt(exception=exc).error("向量检索失败 question={}", question)
        writer(Step(message="正在检索相关片段", status=StepStatus.ERROR, error=str(exc)))
        return Command(goto=END)

    # 2. 加权 RRF 融合：得分 = Σ 权重 / (RRF_K + 名次)，名次从 1 起计
    hits: dict[int | str, ChunkHit] = {}
    scores: dict[int | str, float] = {}
    for results, weight in (
            (question_hits, QUESTION_WEIGHT),
            (hypothesis_hits, HYPOTHESIS_WEIGHT),
    ):
        for rank, hit in enumerate(results, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + weight / (RRF_K + rank)
            hits.setdefault(hit.id, hit)

    # 3. 转成本地 chunk 类型（带上融合得分）并按得分从高到低排序
    local_chunks = sorted(
        (_build_local_chunk(hit, scores[hit.id]) for hit in hits.values()),
        key=lambda item: item.score,
        reverse=True,
    )

    logger.info(
        "向量检索完成 question={} question_hits={} hypothesis_hits={} local_chunks={} top_scores={}",
        question, len(question_hits), len(hypothesis_hits), len(local_chunks),
        [round(item.score, 4) for item in local_chunks[:3]],
    )
    writer(Step(message="正在检索相关片段", status=StepStatus.SUCCESS))
    return Command(update={
        "local_chunks": local_chunks,
        "messages": [AIMessage(content=f"向量检索，召回 {len(local_chunks)} 个本地片段")],
    })


def _build_local_chunk(hit: ChunkHit, score: float) -> LocalChunk:
    """把 milvus 的检索命中转换成本地的 :class:`LocalChunk`。

    状态与后续节点只使用 :class:`~plugins.agents.retrieval.schemas.LocalChunk`，
    不复用 milvus 的检索结果类型，插件侧的类型调整不会渗透进流水线。

    Args:
        hit: :meth:`~plugins.milvus.Milvus.find_chunks` 返回的单条命中。
        score: 该 chunk 的 RRF 融合得分（milvus 的原始 ``distance`` 不保留）。

    Returns:
        带融合得分的本地 chunk。
    """
    return LocalChunk(
        id=hit.id,
        score=score,
        title=hit.title,
        content=hit.content,
        file_name=hit.file_name,
        original_file_url=hit.original_file_url,
    )


async def search_online(state: State, runtime: Runtime[Context]) -> Command:
    """联网搜索，把结果写入 ``state.web_chunks``。

    搜索走运行时上下文注入的 Tavily 客户端（``Context.tavily_client``，见
    :class:`~plugins.agents.retrieval.schemas.Context`）：以改写后的问题调用
    :meth:`~tavily.AsyncTavilyClient.search`，条数取
    :data:`config.TAVILY_MAX_RESULTS`、深度取 :data:`config.TAVILY_SEARCH_DEPTH`；
    响应中 ``results`` 的每一项规范成
    :class:`~plugins.agents.retrieval.schemas.WebChunk`（兼容 ``WebChunk`` 实例与含
    ``title`` / ``content``（缺失取 ``snippet``）/ ``url``（缺失取 ``link``）的映射，
    无法识别的项跳过并告警）。

    Args:
        state: 流水线状态，使用其中的 ``rewritten_question`` 与 ``question``。
        runtime: LangGraph 运行时——``context.tavily_client`` 取搜索客户端，
            ``stream_writer`` 上报步骤进度。

    Returns:
        ``Command``：检索成功时以 ``web_chunks`` 字段回传结果列表；调用失败时跳转尾节点并
        上报 ``error`` 步骤。
    """
    writer = runtime.stream_writer

    # 检索依据为改写后的问题；改写缺失（未执行或失败）时回退到原问题
    query = state.rewritten_question or state.question
    logger.info("联网搜索开始 question={}", query)
    writer(Step(message="正在联网搜索"))

    try:
        response = await runtime.context.tavily_client.search(
            query,
            max_results=config.TAVILY_MAX_RESULTS,
            search_depth=config.TAVILY_SEARCH_DEPTH,
        )
        web_chunks: list[WebChunk] = []
        for item in (response or {}).get("results") or []:
            if isinstance(item, WebChunk):
                web_chunks.append(item)
                continue
            if not isinstance(item, Mapping):
                logger.warning("联网搜索返回项无法识别 type={}", type(item).__name__)
                continue
            web_chunks.append(WebChunk(
                title=str(item.get("title") or ""),
                content=str(item.get("content") or item.get("snippet") or ""),
                url=str(item.get("url") or item.get("link") or ""),
            ))
    except Exception as exc:
        logger.opt(exception=exc).error("联网搜索失败 question={}", query)
        writer(Step(message="正在联网搜索", status=StepStatus.ERROR, error=str(exc)))
        return Command(goto=END)

    logger.info("联网搜索完成 question={} results={}", query, len(web_chunks))
    writer(Step(message="正在联网搜索", status=StepStatus.SUCCESS))
    return Command(update={
        "web_chunks": web_chunks,
        "messages": [AIMessage(content=f"联网搜索，检索到 {len(web_chunks)} 条网页片段")],
    })


async def rerank(state: State, runtime: Runtime[Context]) -> Command:
    """把网络片段与本地 chunk 一起重排，按断崖式阈值截断后融合成 ``state.chunks``。

    候选为 ``state.local_chunks``（向量检索结果）与 ``state.web_chunks``（联网搜索结果），
    正文按「本地在前、网络在后」的顺序一次性提交
    :class:`~plugins.agents.rerank.RerankClient`（连接信息取运行时上下文的
    ``Context.rerank``），以改写后的问题（缺失时回退 ``state.question``）为查询，得到每条
    候选 0~1 的相关度；相关度高的在前排序，再按
    **固定上下限 + 断崖式阈值**（数量上限 :data:`RERANK_MAX_CHUNKS`、下限
    :data:`RERANK_MIN_CHUNKS`；从上往下比较相邻两项，跌幅超过 :data:`RERANK_MAX_GAP`
    或超过前一项分数的 :data:`RERANK_MAX_GAP_RATIO` 即在该处截断）截断；保留的候选统一
    映射成 :class:`~plugins.agents.retrieval.schemas.Chunk`（本地块 ``source="local"``、
    带 ``id`` 与 ``url`` / ``filename``（分别取自 ``original_file_url`` / ``file_name``），网络片段
    ``source="online"``、带 ``url``）后写入 ``state.chunks``。

    Args:
        state: 流水线状态，使用其中的 ``local_chunks``、``web_chunks``、
            ``rewritten_question`` 与 ``question``。
        runtime: LangGraph 运行时（携带 :class:`~plugins.agents.retrieval.schemas.Context`
            上下文，模型 / 重排连接信息 / 搜索客户端等依赖按需从 ``runtime.context`` 取），
            步骤进度由 ``stream_writer`` 上报。

    Returns:
        ``Command``：成功时以 ``chunks`` 字段回传融合后的结果，并在完成步骤的 ``chunks``
        字段里一并带上（便于流式下发与评测收集）；两路候选都为空时返回空 ``Command``
        （不改状态）；调用失败或未返回结果时跳转尾节点并上报 ``error`` 步骤。
    """
    writer = runtime.stream_writer

    question = state.rewritten_question or state.question
    local_chunks = list(state.local_chunks or [])
    web_chunks = list(state.web_chunks or [])
    if not local_chunks and not web_chunks:
        logger.info("重排跳过：没有可重排的候选 question={}", question)
        writer(Step(message="正在重排序候选片段", status=StepStatus.SUCCESS))
        return Command(update={
            "messages": [AIMessage(content="重排序候选片段，没有可保留的片段")]
        })

    logger.info(
        "重排开始 question={} local_chunks={} web_chunks={}",
        question, len(local_chunks), len(web_chunks),
    )
    writer(Step(message="正在重排序候选片段"))

    # 本地在前、网络在后：重排返回的 index 即这里的下标
    documents = [chunk.content for chunk in local_chunks]
    documents += [chunk.content for chunk in web_chunks]
    try:
        # 连接信息（地址 / 模型名 / Key / 任务说明字段名）取自运行时上下文
        results = await rerank_client.rerank(runtime.context.rerank, question, documents)
    except Exception as exc:
        logger.opt(exception=exc).error("重排失败 question={}", question)
        writer(Step(message="正在重排序候选片段", status=StepStatus.ERROR, error=str(exc)))
        return Command(goto=END)

    if not results:
        logger.error("重排失败：未返回任何结果 question={} documents={}", question, len(documents))
        writer(Step(message="正在重排序候选片段", status=StepStatus.ERROR, error="重排未返回结果"))
        return Command(goto=END)

    # 候选就地映射成统一的 Chunk 并按相关度降序：本地块的 id / original_file_url /
    # file_name 落到 id / url / filename，网络片段只带 url；下标越界的项跳过并告警（相关度
    # 只用于排序与截断、不写进 Chunk）
    ranked: list[tuple[float, Chunk]] = []
    for result in results:
        index = result.index
        if not 0 <= index < len(documents):
            logger.warning("重排返回下标越界，已跳过 index={} documents={}", index, len(documents))
            continue
        if index < len(local_chunks):
            chunk = local_chunks[index]
            ranked.append((result.score, Chunk(
                title=chunk.title,
                content=chunk.content,
                source="local",
                id=chunk.id,
                url=chunk.original_file_url,
                filename=chunk.file_name,
            )))
        else:
            chunk = web_chunks[index - len(local_chunks)]
            ranked.append((result.score, Chunk(
                title=chunk.title,
                content=chunk.content,
                source="online",
                url=chunk.url,
            )))
    ranked.sort(key=lambda item: item[0], reverse=True)

    # 固定上下限 + 断崖式阈值：数量先夹到 RERANK_MAX_CHUNKS 以内，再从最高分往下比较相邻
    # 两项，跌幅超过 RERANK_MAX_GAP（绝对）或前一项分数的 RERANK_MAX_GAP_RATIO（相对）即为
    # 断崖，在该处截断（循环从 RERANK_MIN_CHUNKS 处起，保证至少保留这么多项）
    keep = min(len(ranked), RERANK_MAX_CHUNKS)
    for index in range(RERANK_MIN_CHUNKS - 1, max(keep - 1, 0)):
        gap = ranked[index][0] - ranked[index + 1][0]
        if gap > RERANK_MAX_GAP or gap > ranked[index][0] * RERANK_MAX_GAP_RATIO:
            keep = index + 1
            break
    chunks = [chunk for _, chunk in ranked[:keep]]

    logger.info(
        "重排完成 question={} reranked={} kept={} sources={} scores={}",
        question, len(ranked), len(chunks),
        [chunk.source for chunk in chunks],
        [round(score, 4) for score, _ in ranked[:keep]],
    )
    writer(Step(message="正在重排序候选片段", status=StepStatus.SUCCESS, chunks=chunks))
    return Command(update={
        "chunks": chunks,
        "messages": [AIMessage(content=f"重排序候选片段，保留 {len(chunks)} 个片段")],
    })


async def generate_answer(state: State, runtime: Runtime[Context]) -> Command:
    """根据问题与融合后的块生成答案，写入 ``state.answer``。

    输入按 :data:`~plugins.agents.retrieval.prompts.ANSWER_GENERATION_PROMPT` 约定的
    JSON 结构一次性提交（问题在改写阶段已补全为可独立理解的问题）：
    ``question`` 为改写后的问题（缺失时回退 ``state.question``），``chunks`` 每项仅含
    ``title`` / ``content``。模型只依据片段作答，并在句末用 ``[序号]`` 标注引用；序号即
    ``chunks`` 下标 + 1，调用方可按 ``state.chunks`` 同序映射回 ``url`` / ``filename``。
    模型取运行时上下文的 ``runtime.context.model``，调用走
    :func:`~plugins.agents.models.invoke`（输出用
    :class:`~plugins.agents.models.TextOutputValidator` 只校验非空）。

    Args:
        state: 流水线状态，使用其中的 ``chunks``、``rewritten_question`` 与 ``question``。
        runtime: LangGraph 运行时（携带 :class:`~plugins.agents.retrieval.schemas.Context`
            上下文，模型 / 重排连接信息 / 搜索客户端等依赖按需从 ``runtime.context`` 取），
            步骤进度由 ``stream_writer`` 上报。

    Returns:
        ``Command``：成功时以 ``answer`` 字段回传答案，并在步骤的 ``answer`` 字段里一并
        带上（便于流式下发）；重试后仍失败时跳转尾节点并上报 ``error`` 步骤。
    """
    writer = runtime.stream_writer

    question = state.rewritten_question or state.question
    chunks = [
        {"title": chunk.title, "content": chunk.content}
        for chunk in state.chunks or []
    ]
    logger.info(
        "答案生成开始 question={} chunks={}",
        question, len(chunks),
    )
    writer(Step(message="正在生成回答"))

    payload = {"question": question, "chunks": chunks}
    answer, error = await invoke(
        runtime.context.model,
        [
            SystemMessage(content=ANSWER_GENERATION_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
        TextOutputValidator,
    )
    if error:
        logger.error("答案生成失败 question={} err={}", question, error)
        writer(Step(message="正在生成回答", status=StepStatus.ERROR, error=error))
        return Command(goto=END)

    answer = answer.strip()
    logger.info(
        "答案生成完成 question={} chars={} answer={}",
        question, len(answer), answer,
    )
    writer(Step(message="正在生成回答", status=StepStatus.SUCCESS, answer=answer))
    return Command(update={
        "answer": answer,
        "messages": [AIMessage(content=f"生成回答，答案为「{answer}」")],
    })
