"""智能体使用的大模型与向量模型定义。

- :data:`ds`：对话大模型（OpenAI 兼容接口，指向 DeepSeek）；
- :class:`QwenEmbeddingModel`：qwen3.7-text-embedding 向量模型（阿里云百炼
  DashScope 原生接口）。一次调用同时产出**稠密 + 稀疏**向量（供 Milvus 混合检索），
  它同时是 LangChain 的 :class:`~langchain_core.embeddings.Embeddings`（稠密向量，
  可直接交给 Ragas 等框架），见 :class:`~plugins.agents.embedding.Embedding`。
"""

from __future__ import annotations

from abc import abstractclassmethod, ABC
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langchain_openai import ChatOpenAI

from config import (
    DEEPSEEK_MODEL_KEY,
    DEEPSEEK_MODEL_NAME,
    DEEPSEEK_MODEL_URL,
    QWEN_EMBEDDING_DIMENSIONS,
    QWEN_EMBEDDING_KEY,
    QWEN_EMBEDDING_NAME,
    QWEN_EMBEDDING_URL,
)
from logger import logger

__all__ = [
    "ds",
    "EmbeddingAPIError",
    "invoke",
    "JSONOutputValidator",
    "QwenEmbeddingModel",
    "qwen_embedding",
    "RerankModel",
    "TextOutputValidator",
]

#: DashScope 原生文本向量端点（相对构造时注入的 ``url``）
QWEN_EMBEDDINGS_PATH = "/services/embeddings/text-embedding/text-embedding"

#: 对外 ``output_type`` → DashScope 原生接口 ``parameters.output_type`` 取值
_OUTPUT_TYPE_MAP = {
    "dense": "dense",
    "sparse": "sparse",
    "both": "dense&sparse",
}


class EmbeddingAPIError(RuntimeError):
    """文本向量接口调用失败：HTTP 状态码非 2xx、业务码非空，或响应条数不匹配。"""

    def __init__(self, message: str, *, code: str = "") -> None:
        super().__init__(message)
        self.code = code


class QwenEmbeddingModel(Embeddings):
    """qwen3.7-text-embedding 向量模型（阿里云百炼 DashScope 原生接口）。

    接口：``POST /services/embeddings/text-embedding/text-embedding``
    （相对构造时注入的 ``url``，默认 ``https://dashscope.aliyuncs.com/api/v1``）。

    请求体形如::

        {
          "model": "qwen3.7-text-embedding",
          "input": {"texts": ["文本A", "文本B"]},
          "parameters": {"dimension": 1024, "output_type": "dense&sparse"}
        }

    响应 ``output.embeddings[]`` 每项含 ``text_index``（对应输入下标）、``embedding``
    （稠密向量）与 ``sparse_embedding``（``[{"index": 7149, "value": 0.829, ...}]``，
    仅 ``sparse`` / ``dense&sparse`` 返回）；本模型按 ``text_index`` 还原顺序，保证
    返回值与输入文本一一对应。

    稀疏向量只有 DashScope 原生接口（``output_type=sparse``）支持，OpenAI 兼容接口
    不提供，故统一走原生接口；稀疏向量以 ``{词表位置索引: 权重}`` 表示（与
    ``plugins/milvus.py`` 的 ``ChunkInserted.sparse_vector`` 一致）。

    :meth:`embed` 的 ``output_type`` 与返回值对应关系——返回二元组，未被请求的一侧
    为 ``None``::

        output_type="dense"   →  (list[list[float]], None)
        output_type="sparse"  →  (None, list[dict[int, float]])
        output_type="both"    →  (list[list[float]], list[dict[int, float]])

    批量约束（官方文档）：``qwen3.7-text-embedding`` 输入字符串列表时单次最多 20
    条、每条最长 128,000 Token；超出批大小时由 :meth:`embed` 自动切分成多批串行
    请求，调用方无需关心分批。

    作为 LangChain :class:`~langchain_core.embeddings.Embeddings`：
    :meth:`embed_documents` / :meth:`embed_query`（同步）与 :meth:`aembed_documents`
    / :meth:`aembed_query`（异步）返回稠密向量，可直接交给 Ragas 等框架使用；Ragas
    实际走异步接口。

    配置（``url`` / ``key`` / ``name`` / ``dimensions``）由构造函数注入，模块级
    :data:`qwen_embedding` 默认取自 :mod:`config` 的 ``QWEN_EMBEDDING_*``
    （``[tool.model.qwen-embedding]``）。

    检索 / 入库流水线里的稠密向量与稀疏向量都出自本模型：节点按 ``output_type="both"``
    调用，一次请求同时拿到两者（见 :class:`~plugins.agents.embedding.Embedding`）。
    """

    #: 单次请求最大文本条数（qwen3.7-text-embedding 批量上限 20 条）
    MAX_BATCH_SIZE = 20

    def __init__(self, url: str, key: str, name: str, dimensions: int) -> None:
        """注入接口配置，后续请求按此提交 DashScope。

        Args:
            url: DashScope 原生接口根地址。
            key: 百炼 API Key。
            name: 模型名（如 ``qwen3.7-text-embedding``）。
            dimensions: 稠密向量维度，需与 Milvus 字段维度一致。
        """
        self._url = url
        self._key = key
        self._name = name
        self._dimensions = dimensions
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    @property
    def client(self) -> httpx.AsyncClient:
        """取异步 HTTP 客户端；未创建时按需创建。

        首次调用时按需创建（调用方也可提前访问本属性预热），避免调用方被迫先手动
        打开客户端；客户端随实例存活到进程退出，不主动关闭。
        """
        client = self._client
        if client is None:
            if not self._key:
                logger.warning(
                    "QWEN_EMBEDDING_KEY 未配置，稀疏向量调用将返回 401。"
                    "请在 pyproject.toml 的 [tool.model.qwen-embedding] 配置 key 后再调用。"
                )
            client = httpx.AsyncClient(
                base_url=self._url,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0),
            )
            self._client = client
            logger.debug("文本向量客户端已创建 url={}", self._url)
        return client

    @property
    def sync_client(self) -> httpx.Client:
        """取同步 HTTP 客户端；未创建时按需创建。

        同步接口（:meth:`embed_documents` / :meth:`embed_query`）走它：同步代码里驱动
        共享的异步客户端（如另起线程跑事件循环）会把客户端绑到临时事件循环上，循环
        一关客户端即失效，因此同步与异步各用一套客户端、互不影响。
        """
        client = self._sync_client
        if client is None:
            if not self._key:
                logger.warning(
                    "QWEN_EMBEDDING_KEY 未配置，稀疏向量调用将返回 401。"
                    "请在 pyproject.toml 的 [tool.model.qwen-embedding] 配置 key 后再调用。"
                )
            client = httpx.Client(
                base_url=self._url,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0),
            )
            self._sync_client = client
            logger.debug("文本向量同步客户端已创建 url={}", self._url)
        return client

    # 不提供 close()：实例是进程级共享单例（:data:`qwen_embedding`），HTTP 连接池随进程
    # 退出由操作系统回收。主动关闭反而会关掉其他调用方（评测流水线等）正在使用的客户端，
    # 且若在别的事件循环里关闭由本循环创建的客户端，还会报事件循环相关错误。

    # ------------------------------------------------------------------ #
    # 稠密 + 稀疏向量
    # ------------------------------------------------------------------ #
    async def embed(
            self,
            texts: list[str],
            output_type: Literal["dense", "sparse", "both"],
    ) -> tuple[list[list[float]] | None, list[dict[int, float]] | None]:
        """把文本列表转成向量，返回 ``(稠密向量, 稀疏向量)`` 二元组。

        超过 :attr:`MAX_BATCH_SIZE` 条时自动分批串行请求，逐批拼接结果，返回值顺序
        与 ``texts`` 严格一致；未被 ``output_type`` 请求的一侧为 ``None``。

        Args:
            texts: 待向量化的文本列表；空列表返回空结果（未请求的一侧仍为
                ``None``）。
            output_type: 输出类型——``"dense"`` 只要稠密向量，``"sparse"`` 只要
                稀疏向量，``"both"`` 两者都要。

        Returns:
            ``(dense, sparse)``：``dense`` 为 ``list[list[float]]``（第 i 项对应
            ``texts[i]``，维度为构造时注入的 ``dimensions``）；``sparse`` 为
            ``list[dict[int, float]]``（第 i 项为 ``{词表位置索引: 权重}``）。

        Raises:
            ValueError: ``output_type`` 不是 ``dense`` / ``sparse`` / ``both``。
            EmbeddingAPIError: 接口返回非 2xx、业务码非空，或单批返回条数与请求
                不一致。
            httpx.HTTPError: HTTP 传输错误（含网络中断、超时）。
        """
        if output_type not in _OUTPUT_TYPE_MAP:
            raise ValueError(
                f"output_type 仅支持 dense / sparse / both，实际为 {output_type!r}"
            )

        wants_dense = output_type != "sparse"
        wants_sparse = output_type != "dense"
        if not texts:
            return (
                [] if wants_dense else None,
                [] if wants_sparse else None,
            )

        dense_vectors = []
        sparse_vectors = []
        total = len(texts)
        for start in range(0, total, self.MAX_BATCH_SIZE):
            batch = texts[start:start + self.MAX_BATCH_SIZE]
            logger.debug(
                "文本向量分批请求 start={}/{} batch={} output_type={}",
                start, total, len(batch), output_type,
            )
            for item in await self._request_batch(batch, output_type):
                if wants_dense:
                    dense_vectors.append(item.get("embedding") or [])
                if wants_sparse:
                    sparse_vectors.append(self._build_sparse_dict(item))

        logger.info(
            "文本向量生成完成 count={} output_type={} dense={} sparse={}",
            total, output_type, len(dense_vectors), len(sparse_vectors),
        )
        return (
            dense_vectors if wants_dense else None,
            sparse_vectors if wants_sparse else None,
        )

    # ------------------------------------------------------------------ #
    # 内部：单批请求与响应解析
    # ------------------------------------------------------------------ #
    def _build_payload(self, texts: list[str], output_type: str) -> dict[str, Any]:
        """拼装单批请求体。"""
        return {
            "model": self._name,
            "input": {"texts": texts},
            "parameters": {
                "dimension": self._dimensions,
                "output_type": _OUTPUT_TYPE_MAP[output_type],
            },
        }

    async def _request_batch(
            self, texts: list[str], output_type: str
    ) -> list[dict[str, Any]]:
        """异步请求单个批次，返回按输入顺序排列的 ``output.embeddings`` 列表。"""
        logger.debug(
            "文本向量请求 model={} batch={} output_type={} dimension={}",
            self._name, len(texts), output_type,
            self._dimensions,
        )
        resp = await self.client.post(
            QWEN_EMBEDDINGS_PATH, json=self._build_payload(texts, output_type)
        )
        return self._parse_items(resp, texts)

    def _request_batch_sync(
            self, texts: list[str], output_type: str
    ) -> list[dict[str, Any]]:
        """同步请求单个批次（走 :attr:`sync_client`）。"""
        logger.debug(
            "文本向量同步请求 model={} batch={} output_type={} dimension={}",
            self._name, len(texts), output_type,
            self._dimensions,
        )
        resp = self.sync_client.post(
            QWEN_EMBEDDINGS_PATH, json=self._build_payload(texts, output_type)
        )
        return self._parse_items(resp, texts)

    def _parse_items(
            self, resp: httpx.Response, texts: list[str]
    ) -> list[dict[str, Any]]:
        """校验响应，返回按 ``text_index`` 还原为输入顺序的 ``output.embeddings``。"""
        if resp.is_error:
            # 错误响应体可能不是 JSON（如网关 502），统一按文本截断记录
            detail = resp.text[:500]
            logger.error(
                "文本向量接口返回错误 status={} body={}",
                resp.status_code, detail,
            )
            raise EmbeddingAPIError(
                f"文本向量接口返回错误 status={resp.status_code} body={detail}",
                code=str(resp.status_code),
            )

        body = resp.json()
        # 业务失败时 HTTP 仍为 200，错误信息落在 code / message（成功时为空）
        code = body.get("code")
        if code:
            logger.error(
                "文本向量接口业务错误 code={} message={}",
                code, body.get("message"),
            )
            raise EmbeddingAPIError(
                f"文本向量接口业务错误 code={code} message={body.get('message')}",
                code=str(code),
            )

        items = (body.get("output") or {}).get("embeddings") or []
        # 服务端可能乱序返回，按 text_index 还原为输入顺序
        items.sort(key=lambda item: item.get("text_index", 0))
        if len(items) != len(texts):
            logger.error(
                "文本向量返回条数不一致 expected={} actual={}",
                len(texts), len(items),
            )
            raise EmbeddingAPIError(
                f"文本向量返回条数不一致 expected={len(texts)} actual={len(items)}"
            )

        logger.debug(
            "文本向量单批完成 count={} usage={}",
            len(items), (body.get("usage") or {}).get("total_tokens"),
        )
        return items

    @staticmethod
    def _build_sparse_dict(item: dict[str, Any]) -> dict[int, float]:
        """把一条响应的 ``sparse_embedding`` 转成 ``{索引: 权重}``。

        Args:
            item: ``output.embeddings[]`` 中的一项，形如
                ``{"text_index": 0, "sparse_embedding": [{"index": 7149, "value": 0.829, ...}]}``。

        Returns:
            ``{词表位置索引: 权重}``；``sparse_embedding`` 缺失或为空时返回空字典。
        """
        return {
            entry["index"]: entry["value"]
            for entry in item.get("sparse_embedding") or []
        }

    # ------------------------------------------------------------------ #
    # LangChain Embeddings 接口（稠密向量）
    # ------------------------------------------------------------------ #
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """同步批量向量化（LangChain ``Embeddings`` 接口）。

        走同步 HTTP 客户端（:attr:`sync_client`），与异步接口各自独立、不占用事件循环；
        Ragas 等异步框架走 :meth:`aembed_documents`，不会命中这里。
        """
        if not texts:
            return []
        dense_vectors = []
        for start in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[start:start + self.MAX_BATCH_SIZE]
            for item in self._request_batch_sync(batch, "dense"):
                dense_vectors.append(item.get("embedding") or [])
        logger.info(
            "文本向量同步生成完成 count={} dense={}", len(texts), len(dense_vectors)
        )
        return dense_vectors

    def embed_query(self, text: str) -> list[float]:
        """同步单条向量化（LangChain ``Embeddings`` 接口）。"""
        vectors = self.embed_documents([text])
        return vectors[0] if vectors else []

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """异步批量向量化（Ragas 等实际走的路径）。"""
        dense, _ = await self.embed(texts, "dense")
        return list(dense or [])

    async def aembed_query(self, text: str) -> list[float]:
        """异步单条向量化。"""
        vectors = await self.aembed_documents([text])
        return vectors[0] if vectors else []


@dataclass()
class RerankModel:
    """重排模型的连接信息，随图上下文传入重排客户端。

    url：完整请求地址。OpenAI 官方协议没有重排端点，「OpenAI 兼容」的各家路径并不统一
        （百炼是 {base}/compatible-api/v1/reranks，硅基流动是 {base}/v1/rerank），故直接给出完整地址；
    name：模型名，如 Qwen/Qwen3-Reranker-8B；
    key：该服务商的 API Key；
    instruct_field：任务说明（instruction）在请求体里的字段名，各家叫法不统一
        （硅基流动为 instruction，百炼为 instruct）；None 表示不发该字段。
    """
    url: str
    name: str
    key: str
    instruct_field: str | None = "instruct"


ds = ChatOpenAI(
    model=DEEPSEEK_MODEL_NAME,
    api_key=DEEPSEEK_MODEL_KEY,
    base_url=DEEPSEEK_MODEL_URL
)

#: 文本向量模型实例：配置取自 ``[tool.model.qwen-embedding]``（阿里云百炼原生接口），
#: 一次调用同时产出稠密 + 稀疏向量，供流水线（Milvus 混合检索）与 Ragas 复用
qwen_embedding = QwenEmbeddingModel(
    url=QWEN_EMBEDDING_URL,
    key=QWEN_EMBEDDING_KEY,
    name=QWEN_EMBEDDING_NAME,
    dimensions=QWEN_EMBEDDING_DIMENSIONS,
)


class JSONOutputValidator(ABC):
    """大模型 JSON 输出的校验接口：各模型自己把模型返回的文本解析成结构化结果。

    校验失败一律抛异常（解析错误、字段缺失、类型不符），调用方据此重试（见 invoke）。
    输出不是 JSON 的调用点用 :class:`TextOutputValidator`（只校验非空）。
    """

    @abstractclassmethod
    def validate_output(cls, text: str) -> Any:
        pass


class TextOutputValidator(JSONOutputValidator):
    """纯文本输出（答案正文、改写后的问题、产品名称等）的校验器：只要求返回非空。

    这类调用点不要求模型返回 JSON，「校验」退化为排除空输出；调用异常与空输出一样可由
    :func:`invoke` 重试，故所有大模型调用都能统一走 :func:`invoke`。
    """

    @classmethod
    def validate_output(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("模型返回内容为空")
        return text


# 校验失败后的重试指令：连同模型的上一版输出一起作为新一轮对话回传，让它据此自我纠正
RETRY_INSTRUCTION = "上一次输出未通过校验，请按下面的错误信息修正后重新输出" \
                    "（只输出符合要求的 JSON，不要任何解释）：\n{error}"


async def invoke(model: BaseChatModel,
                 messages: list[AnyMessage],
                 validator: type[JSONOutputValidator],
                 max_retry: int = 3,
                 **kwargs) -> tuple[Any, str | None]:
    """调用大模型并校验其输出，校验不过就带上错误信息重试，超过上限返回错误文本而不抛异常。

    :param model: 大模型实例
    :param messages: 本轮对话消息
    :param validator: 输出校验器（见 JSONOutputValidator），负责把模型返回的文本解析成结构化结果
    :param max_retry: 最多尝试几次（含首次），至少为 1
    :param kwargs: 透传给 model.ainvoke 的参数，如 max_tokens、extra_body
    :return: (校验后的输出, 错误文本)：成功时错误文本为空串；达到重试上限时输出为 None，
        错误文本为最后一次的失败原因（调用异常或校验异常）

    重试覆盖两类失败：调用本身抛异常（限流、超时等），以及返回文本没过校验。
    后者会把模型的上一版输出与失败原因（见 RETRY_INSTRUCTION）追加进对话，下一轮模型据此纠正；
    追加只发生在本地副本上，不改动调用方传入的消息列表。
    调用方拿到的永远是一对返回值，不必再自己 try / except；取消等 BaseException 不在此吞掉，直接向上抛。
    """
    error = ""
    # 本地的对话副本：校验失败时往里追加「上一版输出 + 错误信息」，调用方的消息列表保持不变
    attempt_messages = list(messages)
    for attempt in range(1, max_retry + 1):
        try:
            response = await model.ainvoke(attempt_messages, **kwargs)
        except Exception as e:  # noqa: BLE001 - 失败原因统一以错误文本返回给调用方
            error = str(e)
            logger.warning(f"invoke: attempt {attempt}/{max_retry} failed: {error}")
            continue
        try:
            return validator.validate_output(response.content), None
        except Exception as e:  # noqa: BLE001 - 校验失败要回传错误信息，故与调用异常分开处理
            error = str(e)
            logger.warning(f"invoke: attempt {attempt}/{max_retry} failed: {error}")
            attempt_messages += [AIMessage(content=response.content),
                                 HumanMessage(content=RETRY_INSTRUCTION.format(error=error))]
    logger.error(f"invoke: retry limit exceeded after {max_retry} attempt(s): {error}")
    return None, error
