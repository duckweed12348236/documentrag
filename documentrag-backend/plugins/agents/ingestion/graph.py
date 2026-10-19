"""文档入库流水线的图定义：把各节点串联成一条可执行的 LangGraph。

节点顺序::

    understand_images → understand_blocks → split → save_chunks

1. :func:`~plugins.agents.ingestion.nodes.understand_images`：为图片生成描述并写回
   ``alt``；
2. :func:`~plugins.agents.ingestion.nodes.understand_blocks`：把表格与列表转写为
   自然语言；
3. :func:`~plugins.agents.ingestion.nodes.split`：按标题与 token 切成 chunk；
4. :func:`~plugins.agents.ingestion.nodes.save_chunks`：把 chunk 向量化写入 Milvus
   ``chunk`` 集合。

节点之间的跳转由**各节点自己**返回的 ``Command(goto=...)`` 决定（见
:mod:`plugins.agents.ingestion.nodes`）：节点执行成功后跳向下一节点，输入不可用时
跳 ``END`` 并上报 ``error`` 步骤，流水线随即结束、后续节点不再执行。因此本模块
只负责按名字注册节点并声明入口（``START`` → :data:`NODES` 中的首个节点），不设置
任何节点之间的静态边；:data:`NODES` 的元组顺序只决定入口，不再决定流转顺序。

用法::

    from plugins.agents.ingestion.graph import Ingestor

    agent = Ingestor()
    async for step in agent.invoke(md_path, file_url):
        print(step.message, step.status)

``State.file_path`` 与 ``State.original_file_url`` 是必填项，由调用方（上传接口 /
任务）传入，其余字段由各节点回传；执行进度由各节点的 ``stream_writer`` 以
:class:`~plugins.agents.ingestion.schemas.Step` 流式产出。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from pathlib import Path

from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from logger import logger
from plugins import Singleton
from plugins.agents.ingestion.nodes import (
    save_chunks,
    split,
    understand_tables,
    understand_images,
)
from plugins.agents.ingestion.schemas import Context, State, Step
from plugins.agents.models import ds, qwen_embedding

#: 节点函数签名：接收流水线状态与运行时，返回状态更新指令
NodeFunc = Callable[[State, Runtime[Context]], Awaitable[Command]]

#: 流水线节点：``(节点名, 节点函数)``，元组顺序即执行顺序
NODES: Sequence[tuple[str, NodeFunc]] = (
    ("understand_images", understand_images),
    ("understand_blocks", understand_tables),
    ("split", split),
    ("save_chunks", save_chunks),
)


__all__ = ["NODES", "Ingestor", "NodeFunc"]


class Ingestor(Singleton):
    """文档入库智能体：进程内单例，对一份 Markdown 文档跑完整条入库流水线。

    流水线在构造时编译一次（``Singleton`` 保证进程内只构造一个实例），
    :meth:`invoke` 以流式方式执行，逐个产出各节点上报的步骤进度。对话大模型与文本
    向量模型以 :class:`~plugins.agents.ingestion.schemas.Context` 注入图运行时，节点从
    ``runtime.context.model`` / ``runtime.context.embedding_model`` 取用、不在模块级
    共享模型实例。
    """

    def __init__(self) -> None:
        builder = StateGraph(State, context_schema=Context)
        for name, node in NODES:
            builder.add_node(name, node)

        # 只声明入口；节点之间怎么走由各节点返回的 Command(goto=...) 决定
        builder.add_edge(START, NODES[0][0])

        self.graph: CompiledStateGraph = builder.compile()

    async def invoke(
            self, file_path: Path, original_file_url: str
    ) -> AsyncGenerator[Step, None]:
        """流式执行入库流水线，逐个产出步骤进度。

        Args:
            file_path: Markdown 文档路径（MinerU 解析产物）。
            original_file_url: 原文链接，随 chunk / 实体记录一并入库。

        Yields:
            各节点通过 ``runtime.stream_writer`` 上报的
            :class:`~plugins.agents.ingestion.schemas.Step`：节点开始为
            ``running``，结束为 ``success`` / ``error``。替换后的 Markdown 由
            流水线覆盖保存到 ``file_path``，chunk 与实体向量写入 Milvus。
        """
        state = State(file_path=file_path, original_file_url=original_file_url)
        context = Context(model=ds, embedding_model=qwen_embedding)
        logger.info("文档入库开始 path={} url={}", file_path, original_file_url)
        async for step in self.graph.astream(
                state, context=context, stream_mode="custom"
        ):
            yield step
        logger.info("文档入库结束 path={}", file_path)
