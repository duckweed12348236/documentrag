from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.utils.json import parse_json_markdown
from pydantic import BaseModel, RootModel

from plugins.agents.models import JSONOutputValidator, QwenEmbeddingModel
from plugins.md_utils import MdImage


@dataclass()
class Context:
    #: 对话大模型：节点一律从 ``runtime.context`` 取，不在模块级共享模型实例
    model: BaseChatModel
    #: 文本向量模型（按 ``[tool.model.qwen-embedding]`` 构造）：同时产出稠密与稀疏向量，
    #: 节点调用时把它传给 :meth:`~plugins.agents.embedding.Embedding.embed`
    embedding_model: QwenEmbeddingModel


class StepStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


@dataclass()
class Step:
    """流水线的一次进度上报：说明当前在执行什么、结果如何。"""

    #: 当前步骤的说明，如「正在切分文档」
    message: str
    status: StepStatus = StepStatus.RUNNING
    #: 失败原因（``status`` 为 ``error`` 时带上）
    error: str | None = None


@dataclass()
class Chunk:
    title: str
    content: str
    header_index: int


@dataclass()
class State:
    #: Markdown 文档路径与原文链接，由调用方在启动流水线时传入
    file_path: Path
    original_file_url: str
    content: str | None = None
    chunks: list[Chunk] | None = None


@dataclass
class ImageTask:
    """待描述的图片：文档命中项 + 图片载荷。

    ``image`` 为 :func:`~plugins.md_utils.find_images` 命中的
    :class:`~plugins.md_utils.MdImage`，其 ``start`` / ``end`` / ``before`` /
    ``after`` 等字段直接用于批量替换与上下文提交。
    """

    id: str
    image: MdImage
    mime: str = ""
    b64: str = ""


def _parse_json_array(text: str) -> Any:
    """解析模型返回的 JSON 数组文本。

    文本可能带 ```json 代码块围栏、也可能在 JSON 前后夹带解释文字，故先截取首个
    ``[`` 到末个 ``]`` 之间的内容，再交给
    :func:`~langchain_core.utils.json.parse_json_markdown`（围栏与轻微语法瑕疵都由它
    兼容）。

    请求侧不绑定任何结构化参数（DeepSeek 的思考模式既不接受 ``tool_choice``，
    也不支持 JSON Output），JSON 结构由提示词约定、由这里完成「解析 + Pydantic 校验」，
    失败时由 :func:`~plugins.agents.models.invoke` 带上错误信息重试。

    Args:
        text: 模型返回的文本。

    Returns:
        解析出的 JSON 数组。

    Raises:
        ValueError: 文本不是合法 JSON（``JSONDecodeError`` 同属 :class:`ValueError`）。
    """
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        text = text[start:end + 1]
    return parse_json_markdown(text)


class ImageDescription(BaseModel):
    content: str
    id: int | str


class ImageDescriptions(RootModel[list[ImageDescription]], JSONOutputValidator):
    """结构化输出兼校验器：与输入图片一一对应的描述列表（根即 JSON 数组）。

    实现 :class:`~plugins.agents.models.JSONOutputValidator` 的 :meth:`validate_output`：
    把模型返回的文本解析并校验成本类型（见 ``_parse_json_array``），解析或校验失败即
    抛异常，交由 :func:`~plugins.agents.models.invoke` 带上错误信息重试。
    """

    @classmethod
    def validate_output(cls, text: str) -> "ImageDescriptions":
        """解析并校验模型输出。

        Args:
            text: 模型返回的文本（JSON 数组，可能带围栏或前后的解释文字）。

        Returns:
            校验后的描述列表。

        Raises:
            ValueError: 不是合法 JSON，或结构与 :class:`ImageDescription` 不符
                （Pydantic 的 ``ValidationError`` 亦属 :class:`ValueError`）。
        """
        return cls.model_validate(_parse_json_array(text))


class BlockTranscription(BaseModel):
    content: str
    id: int | str


class BlockTranscriptions(RootModel[list[BlockTranscription]], JSONOutputValidator):
    """结构化输出兼校验器：与输入表格/列表一一对应的转写列表（根即 JSON 数组）。

    实现 :class:`~plugins.agents.models.JSONOutputValidator` 的 :meth:`validate_output`：
    把模型返回的文本解析并校验成本类型（见 ``_parse_json_array``），解析或校验失败即
    抛异常，交由 :func:`~plugins.agents.models.invoke` 带上错误信息重试。
    """

    @classmethod
    def validate_output(cls, text: str) -> "BlockTranscriptions":
        """解析并校验模型输出。

        Args:
            text: 模型返回的文本（JSON 数组，可能带围栏或前后的解释文字）。

        Returns:
            校验后的转写列表。

        Raises:
            ValueError: 不是合法 JSON，或结构与 :class:`BlockTranscription` 不符
                （Pydantic 的 ``ValidationError`` 亦属 :class:`ValueError`）。
        """
        return cls.model_validate(_parse_json_array(text))
