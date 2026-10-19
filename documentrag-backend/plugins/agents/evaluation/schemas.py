from dataclasses import dataclass, field
from enum import StrEnum

from langchain_openai.chat_models.base import BaseChatOpenAI
from ragas.metrics import BaseMetric
from ragas.testset import TestsetGenerator

from plugins.agents.retrieval.graph import Retriever
from plugins.agents.retrieval.schemas import Chunk, LocalChunk, WebChunk


class StepStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


@dataclass()
class Step:
    message: str
    status: StepStatus = StepStatus.RUNNING
    error: str | None = None


@dataclass()
class DatasetItemInfo:
    """数据集中单条测试用例（对应 ORM 模型 :class:`~models.dataset.DatasetItem`）。"""

    question: str
    reference: str
    reference_contexts: list[str]


@dataclass()
class ResultDetailInfo:
    """单条用例的检索情况与评测得分（对应 ORM 模型 :class:`~models.dataset.ResultDetail`）。

    ``scores`` 由评测节点跑完指标后回填（Ragas 指标名 → 该用例得分，失败的指标不在其中）；
    未评测时为空字典。
    """

    question: str
    answer: str
    #: 该条目检索到的上下文列表：即发给大模型的 JSON 字段名、Ragas 指标的形参名，也是
    #: :class:`~models.dataset.ResultDetail` 的同名字段
    retrieved_contexts: list[str]
    #: 该用例各指标得分：Ragas 指标名 → 得分（失败的指标不在其中）
    scores: dict[str, float] = field(default_factory=dict)


@dataclass()
class ResultInfo:
    """数据集评测报告（对应 ORM 模型 :class:`~models.dataset.Result`）：
    逐条用例的情况、整体汇总与大模型解读。
    """

    #: 各项用例的情况：逐条收集（检索失败的条目不在其中），每条含该条目的分项得分
    details: list[ResultDetailInfo]
    #: 评测的总体情况：Ragas 指标名 → 全部条目的平均分
    summary: dict[str, float]
    #: 大模型对本次评测的解读（Markdown 报告正文）
    content: str


@dataclass()
class State:
    size: int
    #: 切分好的文档片段（正文文本）：由
    #: :meth:`~plugins.agents.evaluation.graph.Evaluator.invoke` 读取 Markdown 并按标题 /
    #: token 切分后写入，生成节点据此在内部组装成文档对象去造用例；只在 ``dataset_id`` 为空
    #: 时被取用一次，数据集落库后 ``dataset_id`` 就填上了，之后（循环评测）不再用它
    content: list[str]
    #: 数据集主键：非空即表示数据集已存在，生成节点据此直接跳到评测节点、绝不重新生成；
    #: 为空时由 :func:`generate_dataset` 用 ``content`` 生成并落库后回填
    dataset_id: int | None = None
    #: 数据集评测报告：逐条用例情况、整体指标均分与大模型解读
    result_info: ResultInfo | None = None


#: 会进入流水线状态、因而需要写进短期记忆（checkpoint）的自定义类型：作为 msgpack
#: 白名单交给 checkpointer 的序列化器（见
#: :meth:`~plugins.agents.evaluation.graph.Evaluator.register`）。langgraph 默认只放行一组
#: 内置安全类型，其余类型反序列化时会告警、未来版本直接拦截——拦截后只会还原成原始字典，
#: 调用方再取 ``.details`` / ``.summary`` 就会出错，故在此显式登记。
#:
#: :class:`State` 本身也要登记：图把本次输入原样存入 ``__start__`` 通道，恢复会话时
#: 同样要按类型还原。检索片段（``Chunk`` / ``LocalChunk`` / ``WebChunk``）也一并放行：
#: 早期版本的逐条用例情况里直接存了 chunk 对象，这些历史 checkpoint 仍会被读取，不放行
#: 就会被拦成原始字典并打 ``Blocked deserialization`` 告警。:class:`DatasetItemInfo`
#: 现在只在评测节点内部使用、不再进状态，保留登记是为了兼容用例列表还落在状态里的历史
#: checkpoint。
CHECKPOINT_TYPES = (
    State,
    DatasetItemInfo,
    ResultDetailInfo,
    ResultInfo,
    Chunk,
    LocalChunk,
    WebChunk,
)


@dataclass()
class Context:
    model: BaseChatOpenAI
    generator: TestsetGenerator
    retriever: Retriever
    #: 评测指标（Ragas 指标实例，需已注入 LLM / embeddings），逐条用例调用其 ``ascore``
    metrics: list[BaseMetric]
