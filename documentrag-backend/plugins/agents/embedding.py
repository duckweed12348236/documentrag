"""文本向量插件：模型在调用时传入，插件自身不绑定默认模型。

模型实现见 :class:`~plugins.agents.models.QwenEmbeddingModel`（阿里云百炼 DashScope 原生
接口）：它一次调用即可产出**稠密 + 稀疏**向量（供 Milvus 混合检索），因此本插件按
``output_type`` 把请求原样转交给调用时传入的模型；模型实例由运行时上下文提供
（``Context.embedding_model``，见
:class:`~plugins.agents.retrieval.schemas.Context` 与
:class:`~plugins.agents.ingestion.schemas.Context`）。

本插件只补齐应用生命周期——以**进程内单例**对外提供服务；HTTP 客户端由模型自己按需创建，
连接池随进程退出回收，不主动关闭。

使用::

    from plugins.agents.embedding import Embedding

    embedder = Embedding()                                              # 进程内单例
    dense, sparse = await embedder.embed(context.embedding_model, ["文本A"], "both")
"""

from __future__ import annotations

from typing import Literal

from plugins import Singleton
from plugins.agents.models import EmbeddingAPIError, QwenEmbeddingModel

__all__ = ["Embedding", "EmbeddingAPIError"]


class Embedding(Singleton):
    """文本向量插件：模型于调用时传入（不持有、也不绑定默认模型）。

    一次调用产出稠密 + 稀疏向量，分批约束与错误处理都在
    :class:`~plugins.agents.models.QwenEmbeddingModel` 内完成。
    """

    async def embed(
            self,
            model: QwenEmbeddingModel,
            texts: list[str],
            output_type: Literal["dense", "sparse", "both"],
    ) -> tuple[list[list[float]] | None, list[dict[int, float]] | None]:
        """把文本列表转成向量，返回 ``(稠密向量, 稀疏向量)`` 二元组。

        分批约束、返回值顺序与异常见
        :meth:`~plugins.agents.models.QwenEmbeddingModel.embed`：本方法只是把请求转交给
        调用时传入的模型。

        Args:
            model: 文本向量模型（取自运行时上下文的 ``Context.embedding_model``）——
                同一个模型既产出稠密向量、也产出稀疏向量。
            texts: 待向量化的文本列表。
            output_type: 输出类型——``"dense"`` 只要稠密向量，``"sparse"`` 只要稀疏向量，
                ``"both"`` 两者都要。

        Returns:
            ``(dense, sparse)``：第 i 项对应 ``texts[i]``，未被 ``output_type`` 请求的
            一侧为 ``None``（``texts`` 为空时返回空列表）。
        """
        return await model.embed(texts, output_type)
