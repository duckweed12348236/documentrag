from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal

from langchain.messages import AnyMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.utils.json import parse_json_markdown
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from tavily import AsyncTavilyClient

from plugins.agents.models import JSONOutputValidator, QwenEmbeddingModel, RerankModel


@dataclass()
class Context:
    #: 对话大模型：节点一律从 ``runtime.context`` 取，不在模块级共享模型实例
    model: BaseChatModel
    #: 文本向量模型（按 ``[tool.model.qwen-embedding]`` 构造）：同时产出稠密与稀疏向量，
    #: 节点调用时把它传给 :meth:`~plugins.agents.embedding.Embedding.embed`
    embedding_model: QwenEmbeddingModel
    #: 重排模型的连接信息：重排节点把它交给重排客户端使用
    rerank: RerankModel
    #: Tavily 联网搜索客户端（由调用方按 ``[tool.tavily].key`` 构造注入）
    tavily_client: AsyncTavilyClient


class StepStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


@dataclass()
class Step:
    """流水线的一次进度上报：说明当前在执行什么节点、结果如何。"""

    #: 当前步骤的说明，如「正在识别用户意图」
    message: str
    status: StepStatus = StepStatus.RUNNING
    #: 失败原因（``status`` 为 ``error`` 时带上）
    error: str | None = None
    #: 答案生成节点的产出：答案正文（成功时带上，便于随步骤流式下发给前端）
    answer: str | None = None
    #: 检索到的块（重排节点的完成步骤带上，含标题 / 来源 URL 等；此前节点为空）
    chunks: list["Chunk"] | None = None


@dataclass()
class WebChunk:
    """联网搜索命中的网页片段。"""

    title: str
    content: str
    url: str


@dataclass()
class LocalChunk:
    """本地向量库（Milvus）检索命中的 chunk。

    独立定义、**不复用** :mod:`plugins.milvus` 的检索结果类型（``ChunkHit``），
    插件的类型调整不会渗透进流水线状态；只保留排序得分 ``score``，milvus 给出的原始
    相似度 ``distance`` 在融合后与最终名次不再对应，故不保留。
    """

    id: int | str
    #: 排序得分：向量检索阶段为 RRF 融合得分（``Σ 权重 / (RRF_K + 名次)``，被两路命中
    #: 时相加），重排阶段被模型相关度（0~1）覆盖
    score: float
    title: str
    content: str
    file_name: str
    original_file_url: str


@dataclass()
class Chunk:
    title: str
    content: str
    source: Literal["online", "local"]
    id: int | str | None = None
    url: str | None = None
    filename: str | None = None


@dataclass()
class State:
    question: str
    rewritten_question: str | None = None
    #: 流水线的消息列表（**追加**语义，由 ``add_messages`` 归约）：调用方把本轮用户问题、
    #: 各节点把本轮结论（如「识别用户意图，用户意图为产品咨询」）逐条追加进来；节点提交给模型
    #: 的是「历史消息 + 最新问题」。注意归约器必须直接挂在字段注解上——写成 ``... | None``
    #: 会让 LangGraph 取不到归约器、退化成整体覆盖。
    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)
    hypothesis: str | None = None
    #: 联网搜索得到的网页片段
    web_chunks: list[WebChunk] | None = None
    #: 向量检索得到的本地 chunk（由 milvus 命中转换而来，不复用其检索结果类型）
    local_chunks: list[LocalChunk] | None = None
    #: 融合后的 chunk
    chunks: list[Chunk] | None = None
    #: 本轮给用户的回答：答案生成节点产出的答案，或意图识别节点产出的反问语句
    answer: str | None = None


#: 会进入流水线状态、因而需要写进短期记忆（checkpoint）的自定义类型：作为 msgpack
#: 白名单交给 checkpointer 的序列化器（见
#: :meth:`~plugins.agents.retrieval.graph.Retriever.register`）。langgraph 默认只放行一组
#: 内置安全类型，其余类型反序列化时会告警、未来版本直接拦截——拦截后只会还原成原始字典，
#: 节点再取 ``.title`` / ``.content`` 就会出错，故在此显式登记。
#:
#: :class:`State` 本身也要登记：图把本次输入原样存入 ``__start__`` 通道，恢复会话时
#: 同样要按类型还原。
CHECKPOINT_TYPES = (State, WebChunk, LocalChunk, Chunk)


class Intention(BaseModel, JSONOutputValidator):
    """意图识别结果：最新问题是否属于产品咨询、询问对象是否明确、要不要反问用户。

    - ``continued``：``True`` 表示最新问题属于产品咨询；
    - ``object_explicit``：询问对象是否已明确——最新问题显式提及，或能由对话历史唯一
      确定（只判定"是否明确"，不输出对象名称本身）；
    - ``clarification``：询问对象不明确时用于反问用户的语句，其余情况为 ``None``。

    同时是 :class:`~plugins.agents.models.JSONOutputValidator` 的实现：:meth:`validate_output`
    把模型返回的文本解析并校验成本类型，失败即抛异常，交由
    :func:`~plugins.agents.models.invoke` 带上错误信息重试。
    """

    continued: bool
    object_explicit: bool = False
    clarification: str | None = None

    @classmethod
    def validate_output(cls, text: str) -> "Intention":
        """解析并校验模型输出。

        模型输出的 JSON 常被 Markdown 围栏包住，故先交给
        :func:`~langchain_core.utils.json.parse_json_markdown` 解析，再交 Pydantic 校验。

        Args:
            text: 模型返回的文本（JSON 对象，可能带围栏或前后的解释文字）。

        Returns:
            校验后的意图识别结果。

        Raises:
            ValueError: 不是合法 JSON，或结构与本模型不符（Pydantic 的
                ``ValidationError`` 亦属 :class:`ValueError`）。
        """
        return cls.model_validate(parse_json_markdown(text))

