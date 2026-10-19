"""文档入库流水线的节点实现。

:func:`understand_images`（图像理解）：

1. 读取 ``state.file_path`` 指向的 Markdown 原文；
2. 用 :func:`~plugins.md_utils.find_images` 解析出全部图片
   （``![alt](url)``，自带原文偏移与前后 100 字上下文），**以下标作为图片 id**；
3. 逐张从本地读取图片（相对路径按 Markdown 所在目录解析，MinerU 产物的
   ``images/xxx.jpg`` 即属此类），转成 base64 data URI；
4. 截取图片位置 **前后各 100 字** 的上下文作为 ``before`` / ``after``；
5. 以 **5 张为一个单位** 调用运行时上下文的对话大模型
   （``runtime.context.model``）：每张图片前各带一条文本块，提交该图片的
   ``id`` / ``before`` / ``after``，形成「一条文本 → 一张图片」交替出现；图片用
   标准 ``image`` 块提交（元数据只能以文本形式提交——既不能挂在图片块的额外字段
   上，langchain 转换时会丢弃，接口也不会把它交给模型，模型看不到 id 便无法原样
   回填，描述也就写不回原文）；调用走
   :func:`~plugins.agents.models.invoke`，输出由
   :class:`~plugins.agents.ingestion.schemas.ImageDescriptions` 自行解析并校验（该模型
   同时是校验器）
   （提示词见 :data:`~plugins.agents.ingestion.prompts.IMAGE_UNDERSTANDING_PROMPT`）；
6. 将返回的描述写入原图片语法的 ``alt`` 位置（即 ``![描述](url)``），
   批量替换交给 :func:`~plugins.md_utils.replace`；随后覆盖保存 Markdown，并通过
   ``Command(update={"content": ...})`` 把替换后的 Markdown 回传给状态。

容错：单张图片读取失败、单个批次调用或解析失败只告警，不阻断整体流程，
未取得描述的图片保持原样。

:func:`understand_blocks`（表格转写）：

1. 优先取 ``state.content``（图像理解后的 Markdown），缺失时回退读取
   ``state.file_path``；
2. 用 :func:`~plugins.md_utils.find_blocks` 解析出全部表格（Markdown 管道表格与
   HTML ``<table>``，自带原文偏移与前后 100 字上下文），**以下标作为 id**；
3. 每 **5 个** 为一个单位调用运行时上下文的对话大模型
   （``runtime.context.model``）：``id`` / ``after``（前文）/ ``before``
   （后文）/ ``content``（块原文）以 JSON 数组提交，调用走
   :func:`~plugins.agents.models.invoke`，输出由
   :class:`~plugins.agents.ingestion.schemas.BlockTranscriptions` 自行解析并校验（该模型
   同时是校验器）
   （提示词见 :data:`~plugins.agents.ingestion.prompts.TABLE_UNDERSTANDING_PROMPT`）；
4. 用 :func:`~plugins.md_utils.replace` 把转写文本批量替换回原文，覆盖保存
   Markdown，并通过 ``Command(update={"content": ...})`` 回传。

容错：单个批次调用或解析失败只告警，未取得转写结果的块保持原样。

:func:`split`（文档切分）：

1. 优先取 ``state.content``（图像理解后的 Markdown），缺失时回退读取
   ``state.file_path``；
2. 交给 :func:`~plugins.md_utils.split_by_header` 先按 Markdown 标题切成章节，
   再对每段正文用 :func:`~plugins.md_utils.split` 按 token 切小块；
3. 每个小块整理为 :class:`~plugins.agents.ingestion.schemas.Chunk`
   （``title`` 所属章节标题、``content`` 小块正文、``header_index`` 标题行号；
   前言部分标题为空串、行号为 0），以 ``Command(update={"chunks": ...})`` 回传。

:func:`save_chunks`（chunk 入库）：

1. 取 ``state.chunks`` 的全部正文，连同运行时上下文的向量模型
   （``Context.embedding_model``）交给 :class:`~plugins.agents.embedding.Embedding`
   生成稠密 + 稀疏向量（模型内部按
   :attr:`~plugins.agents.models.QwenEmbeddingModel.MAX_BATCH_SIZE` 自动分批）；
2. 连同 ``state.original_file_url``、Markdown 文件名等元数据（实体名字段落空值）组装成
   chunk 记录，经 :meth:`~plugins.milvus.Milvus.add_chunks` 写入 ``chunk`` 集合。

状态：``file_path``（Markdown 路径）与 ``original_file_url``（原文链接）
由调用方在启动流水线时传入，其余字段由各节点回传。

跳转：节点之间的流转由各节点返回的 ``Command(goto=...)`` 决定——执行成功后跳向
下一个节点（节点名与 :mod:`plugins.agents.ingestion.graph` 中注册的一致），输入
不可用或调用失败时跳 ``END`` 结束整条流水线；图里因此不需要任何静态边。
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import re
from pathlib import Path

import aiofiles
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    HumanMessage,
    ImageContentBlock,
    SystemMessage,
    TextContentBlock,
)
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command

from logger import logger
from plugins.agents.ingestion.prompts import (
    TABLE_UNDERSTANDING_PROMPT,
    IMAGE_UNDERSTANDING_PROMPT,
)
from plugins.agents.ingestion.schemas import (
    BlockTranscriptions,
    Chunk,
    Context,
    ImageDescriptions,
    ImageTask,
    State,
    Step,
    StepStatus,
)
from plugins.agents.models import invoke
from plugins.agents.embedding import Embedding
from plugins.json import dump as json_dump
from plugins.md_utils import (
    MdImage,
    MdTable,
    find_tables,
    find_images,
    replace,
    split as md_split,
    split_by_header,
)
from plugins.milvus import ChunkInserted, Milvus

#: 上下文窗口：图片位置前后各截取的字符数
CONTEXT_WINDOW = 100
#: 单次大模型调用提交的块数量（图片；表格转写同）
BATCH_SIZE = 5
#: 无法从扩展名推断 MIME 时的兜底类型
_DEFAULT_MIME = "image/png"


async def understand_images(state: State, runtime: Runtime[Context]) -> Command:
    """为 Markdown 中的图片生成描述并写回 ``alt``。

    Args:
        state: 流水线状态，使用其中的 ``file_path``。
        runtime: LangGraph 运行时（携带 :class:`~plugins.agents.ingestion.schemas.Context`
            上下文，对话大模型等依赖按需从 ``runtime.context`` 取），步骤进度由
            ``stream_writer`` 上报。

    Returns:
        ``Command``：成功时以 ``content`` 字段回传替换描述后的 Markdown
        （已同步覆盖保存到 ``state.file_path``）；Markdown 缺失时跳转尾节点。
    """
    writer = runtime.stream_writer

    file_path = state.file_path
    if not file_path.is_file():
        logger.error("图像理解跳过：Markdown 不存在 path={}", file_path)
        writer(Step(message="正在理解图片", status=StepStatus.ERROR, error=f"Markdown 文件不存在：{file_path}"))
        return Command(goto=END)

    async with aiofiles.open(file_path, encoding="utf-8") as file:
        content = await file.read()
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # 命中项（含原文偏移与前后文）由 md_utils 解析给出，以下标作为图片 id
    tasks = [
        ImageTask(id=str(index), image=image)
        for index, image in enumerate(find_images(content, context=CONTEXT_WINDOW))
    ]
    if not tasks:
        logger.info("图像理解跳过：Markdown 中未匹配到图片 path={}", file_path)
        await _save_markdown(file_path, content)
        writer(Step(message="正在理解图片", status=StepStatus.SUCCESS))
        return Command(update={"content": content}, goto="understand_blocks")

    logger.info("图像理解开始 path={} images={}", file_path, len(tasks))
    writer(Step(message="正在理解图片"))

    # 1. 读取图片并转 base64（IO 密集，并发执行）
    await _attach_payloads(tasks, file_path.parent)

    # 2. 分批调用大模型（每批 BATCH_SIZE 张）
    loaded = [task for task in tasks if task.b64]
    descriptions = {}
    if loaded:
        batches = [
            loaded[i:i + BATCH_SIZE] for i in range(0, len(loaded), BATCH_SIZE)
        ]
        logger.info(
            "图像理解分批调用大模型 path={} batches={} size={}",
            file_path, len(batches), BATCH_SIZE,
        )
        responses = await asyncio.gather(
            *(_describe_batch(runtime.context.model, batch) for batch in batches),
            return_exceptions=True,
        )
        for batch, res in zip(batches, responses):
            if isinstance(res, BaseException):
                logger.opt(exception=res).error(
                    "图像理解批次失败 ids={}",
                    [task.id for task in batch],
                )
                continue
            descriptions.update(res)

    # 3. 描述写回 alt 并落盘
    new_content = _apply_descriptions(content, tasks, descriptions)
    await _save_markdown(file_path, new_content)
    logger.info(
        "图像理解完成 path={} images={} described={}",
        file_path, len(tasks), len(descriptions),
    )
    writer(Step(message="正在理解图片", status=StepStatus.SUCCESS))
    return Command(update={"content": new_content}, goto="understand_blocks")


# ---------------------------------------------------------------------- #
# 内部：读写文档 / 取图 / 调用 / 回写
# ---------------------------------------------------------------------- #
async def _save_markdown(file_path: Path, content: str) -> None:
    """把 Markdown 内容覆盖写回文档。

    改写 Markdown 的节点（图像理解、表格与列表转写）都以本函数落盘，保证文档
    内容与回传给状态的 ``content`` 一致；文件不存在时会新建。
    """
    async with aiofiles.open(file_path, "w", encoding="utf-8") as file:
        await file.write(content)


async def _attach_payloads(tasks: list[ImageTask], base_dir: Path) -> None:
    """并发从本地读取每张图片并填充 ``mime`` / ``b64``；失败仅告警不抛出。"""
    results = await asyncio.gather(
        *(_load_image(task.image.src, base_dir) for task in tasks),
        return_exceptions=True,
    )
    for task, result in zip(tasks, results):
        if isinstance(result, BaseException):
            logger.error("图片读取失败 path={} err={}", task.image.src, result)
            continue
        data, mime = result
        task.mime = mime
        task.b64 = base64.b64encode(data).decode("ascii")


async def _load_image(source: str, base_dir: Path) -> tuple[bytes, str]:
    """从本地读取图片字节与 MIME 类型。

    图片只从本地读取：``source`` 为绝对路径时直接使用，否则按 Markdown 所在
    目录解析为相对路径（MinerU 产物的 ``images/xxx.jpg`` 即属后者）。

    Args:
        source: 图片路径。
        base_dir: Markdown 所在目录，用于解析相对路径。

    Returns:
        ``(图片字节, MIME 类型)``。

    Raises:
        OSError: 图片不存在或不可读。
    """
    path = Path(source)
    if not path.is_absolute():
        path = base_dir / path
    async with aiofiles.open(path, "rb") as file:
        data = await file.read()
    return data, _guess_mime(path.name)


def _guess_mime(name: str) -> str:
    """按扩展名推断 MIME，无法推断时回退 ``image/png``。"""
    mime, _ = mimetypes.guess_type(name)
    return mime or _DEFAULT_MIME


async def _describe_batch(model: BaseChatModel, tasks: list[ImageTask]) -> dict[str, str]:
    """把一批图片连同元数据提交大模型，返回 ``{id: 描述}``。

    元数据（``id`` / ``before`` / ``after``）以 JSON 对象放在**对应图片之前**的
    文本块里，形成「一条文本 → 一张图片」交替出现——与提示词描述的输入结构一致。
    注意两点：元数据只能以文本形式提交，挂在图片块的额外字段上会被 langchain
    转换时丢弃（转换只保留 ``type`` / ``image_url``），即便进了请求体接口也不会
    把它交给模型，模型看不到 id 就无法原样回填、描述写不回原文；图片块的
    ``base64`` 必须是**裸 base64**，完整 data URI 会被 langchain 再拼一次前缀而非法。

    模型返回的 JSON 数组由
    :class:`~plugins.agents.ingestion.schemas.ImageDescriptions` 自行解析并校验（它同时
    是校验器）；调用统一走 :func:`~plugins.agents.models.invoke`，调用异常或校验失败时由
    它重试。

    Args:
        model: 对话大模型（取运行时上下文的 ``Context.model``）。
        tasks: 同一批次（≤ :data:`BATCH_SIZE` 张）的图片任务，均已填充
            ``mime`` / ``b64``。

    Returns:
        id 到描述文本的映射；重试后仍拿不到可用结果时为空字典。
    """
    # 交替提交：该图片的元数据文本 → 该图片，元数据与图片一一对应
    content: list[ImageContentBlock | TextContentBlock] = []
    for task in tasks:
        content.append({
            "type": "text",
            "text": json_dump({
                "id": task.id,
                "before": task.image.before,
                "after": task.image.after,
            }),
        })
        # 标准 image 块：``base64`` 必须是**裸 base64**，``data:`` 前缀由 langchain
        # 转换时按 ``mime_type`` 补上；这里若填完整的 data URI，会被拼成两遍前缀，
        # 接口按非法 base64 直接 400（Invalid base64 data）。
        content.append({
            "type": "image",
            "base64": task.b64,
            "mime_type": task.mime,
        })
    descriptions, error = await invoke(
        model,
        [
            SystemMessage(content=IMAGE_UNDERSTANDING_PROMPT),
            HumanMessage(content=content)
        ],
        ImageDescriptions,
    )
    if error:
        logger.error(
            "图像描述失败 ids={} err={}",
            [task.id for task in tasks], error,
        )
        return {}

    # 优先按 id 匹配（提示词要求模型原样回填 id）
    by_id = {str(item.id): item.content.strip() for item in descriptions.root}
    matched = {task.id: by_id[task.id] for task in tasks if by_id.get(task.id)}
    if matched or len(descriptions.root) != len(tasks):
        if not matched:
            # 条数也对不上，无法安全回填：告警并放弃本批（描述会原样保留图片）
            logger.warning(
                "图像描述与图片对不上，丢弃本批 ids={} returned={}",
                [task.id for task in tasks], [str(item.id) for item in descriptions.root],
            )
        return matched

    # 一条 id 都对不上而条数一致：按顺序一一对应回填，避免整批描述被丢弃
    logger.warning(
        "图像描述 id 全部未匹配，按顺序回填 ids={} returned={}",
        [task.id for task in tasks], [str(item.id) for item in descriptions.root],
    )
    return {
        task.id: item.content.strip()
        for task, item in zip(tasks, descriptions.root)
        if item.content.strip()
    }


def _apply_descriptions(
        content: str,
        tasks: list[ImageTask],
        descriptions: dict[str, str],
) -> str:
    """把描述批量写回图片语法的 ``alt`` 位置。

    替换区间取自 :func:`~plugins.md_utils.find_images` 给出的原文偏移，统一交给
    :func:`~plugins.md_utils.replace` 从后往前落笔；没有描述的图片不产生替换项，
    原样保留。
    """
    replacements = []
    for task in tasks:
        description = descriptions.get(task.id)
        if not description:
            continue
        replacements.append(
            (task.image.start, task.image.end, _render_image(task.image, description))
        )
    return replace(content, replacements)


def _render_image(image: MdImage, description: str) -> str:
    """把描述写进图片的 ``alt``，URL 与标题保持原文不变。

    :class:`~plugins.md_utils.MdImage` 的 ``raw`` 形如 ``![alt](src "title")``，
    首个 ``](`` 之后即 URL 与标题；直接沿用原文尾部，可避免回填解析归一化后的
    ``image.src``（含空格或百分号编码的地址会与原文不一致，甚至破坏语法）。
    """
    return f"![{_sanitize_alt(description)}" + image.raw[image.raw.index("](", 2):]


def _sanitize_alt(description: str) -> str:
    """清洗描述文本，使其可作为 Markdown 图片的 alt 内容。"""
    text = re.sub(r"\s+", " ", description).strip()
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


async def understand_tables(state: State, runtime: Runtime[Context]) -> Command:
    """把 Markdown 中的表格转写为自然语言并替换回原文。

    命中项由 :func:`~plugins.md_utils.find_blocks` 解析给出（Markdown 管道表格与
    HTML ``<table>``，均带原文偏移与前后 100 字上下文），**以下标作为 id**；
    每 :data:`BATCH_SIZE` 个一批提交大模型转写（提示词见
    :data:`~plugins.agents.ingestion.prompts.TABLE_UNDERSTANDING_PROMPT`），
    结果经 :func:`~plugins.md_utils.replace` 批量替换回原文。

    Args:
        state: 流水线状态，优先使用 ``content``，为空时回退读取 ``file_path``。
        runtime: LangGraph 运行时（携带 :class:`~plugins.agents.ingestion.schemas.Context`
            上下文，对话大模型等依赖按需从 ``runtime.context`` 取），步骤进度由
            ``stream_writer`` 上报。

    Returns:
        ``Command``：以 ``content`` 字段回传替换后的 Markdown（已同步覆盖保存到
        ``state.file_path``）；无表格时原样回传；文档缺失时跳转尾节点。
    """
    writer = runtime.stream_writer

    file_path = state.file_path
    content = state.content
    if content is None:
        if not file_path.is_file():
            logger.error("块转写跳过：Markdown 不存在 path={}", file_path)
            writer(Step(message="正在转写表格", status=StepStatus.ERROR, error=f"Markdown 文件不存在：{file_path}"))
            return Command(goto=END)
        async with aiofiles.open(file_path, encoding="utf-8") as file:
            content = await file.read()
        content = content.replace("\r\n", "\n").replace("\r", "\n")

    blocks = find_tables(content, context=CONTEXT_WINDOW)
    if not blocks:
        logger.info("块转写跳过：Markdown 中未匹配到表格 path={}", file_path)
        await _save_markdown(file_path, content)
        writer(Step(message="正在转写表格", status=StepStatus.SUCCESS))
        return Command(update={"content": content}, goto="split")

    logger.info("块转写开始 path={} blocks={}", file_path, len(blocks))
    writer(Step(message="正在转写表格"))

    # 每 BATCH_SIZE 个一批调用大模型（批次之间并发）
    items = list(enumerate(blocks))
    batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    transcriptions = {}
    responses = await asyncio.gather(
        *(_transcribe_batch(runtime.context.model, batch) for batch in batches),
        return_exceptions=True,
    )
    for batch, res in zip(batches, responses):
        if isinstance(res, BaseException):
            logger.opt(exception=res).error(
                "块转写批次失败 ids={}",
                [str(index) for index, _block in batch],
            )
            continue
        transcriptions.update(res)

    new_content = _apply_transcriptions(content, blocks, transcriptions)
    await _save_markdown(file_path, new_content)
    logger.info(
        "块转写完成 path={} blocks={} transcribed={}",
        file_path, len(blocks), len(transcriptions),
    )
    writer(Step(message="正在转写表格", status=StepStatus.SUCCESS))
    return Command(update={"content": new_content}, goto="split")


async def _transcribe_batch(
        model: BaseChatModel, items: list[tuple[int, MdTable]]
) -> dict[str, str]:
    """把一批表格提交大模型转写，返回 ``{id: 转写文本}``。

    每条元素按提示词约定给出 ``id`` / ``after`` / ``before`` / ``content``：
    ``content`` 为块原文，``after`` 为**前文**、``before`` 为**后文**（提示词
    中 ``after`` / ``before`` 的语义与英文字面含义相反，见
    :data:`~plugins.agents.ingestion.prompts.TABLE_UNDERSTANDING_PROMPT`）。

    模型返回的 JSON 数组由
    :class:`~plugins.agents.ingestion.schemas.BlockTranscriptions` 自行解析并校验（它同时
    是校验器）；调用统一走 :func:`~plugins.agents.models.invoke`，调用异常或校验失败时由
    它重试。

    Args:
        model: 对话大模型（取运行时上下文的 ``Context.model``）。
        items: 同一批次（≤ :data:`BATCH_SIZE` 个）的 ``(下标, 命中项)``。

    Returns:
        id 到转写文本的映射；重试后仍拿不到可用结果时为空字典。
    """
    payload = [
        {
            "id": str(index),
            "after": block.before,
            "before": block.after,
            "content": block.raw,
        }
        for index, block in items
    ]
    transcriptions, error = await invoke(
        model,
        [
            SystemMessage(content=TABLE_UNDERSTANDING_PROMPT),
            HumanMessage(content=json_dump(payload)),
        ],
        BlockTranscriptions,
    )
    if error:
        logger.error(
            "块转写失败 ids={} err={}",
            [str(index) for index, _block in items], error,
        )
        return {}
    return {
        str(item.id): item.content.strip()
        for item in transcriptions.root
        if item.content.strip()
    }


def _apply_transcriptions(
        content: str,
        blocks: list[MdTable],
        transcriptions: dict[str, str],
) -> str:
    """把转写文本批量替换回原文对应位置。

    替换区间取自 :func:`~plugins.md_utils.find_blocks` 给出的原文偏移（互不重叠
    的表格），统一交给 :func:`~plugins.md_utils.replace` 从后往前落笔；没有转写
    结果的块不产生替换项，原样保留。
    """
    replacements = []
    for index, block in enumerate(blocks):
        text = transcriptions.get(str(index))
        if not text:
            continue
        replacements.append((block.start, block.end, text))
    return replace(content, replacements)


async def split(state: State, runtime: Runtime[Context]) -> Command:
    """按 Markdown 标题切分，再把章节正文切小，得到 :class:`Chunk` 列表。

    切分用 :func:`~plugins.md_utils.split_by_header` 与
    :func:`~plugins.md_utils.split`：先按标题切成章节（正文不含标题行，标题与
    标题行号落到 ``title`` / ``header_index``；文档开头的前言部分 ``title`` 为
    空串、``header_index`` 为 0），再对每段正文按 token 分小块，小块沿用所属
    章节的标题与标题行号。

    Args:
        state: 流水线状态，优先使用 ``content``，为空时回退读取 ``file_path``。
        runtime: LangGraph 运行时（携带 :class:`~plugins.agents.ingestion.schemas.Context`
            上下文，对话大模型等依赖按需从 ``runtime.context`` 取），步骤进度由
            ``stream_writer`` 上报。

    Returns:
        ``Command``：以 ``chunks`` 字段回传切分结果；无内容可切分时跳转尾节点。
    """
    writer = runtime.stream_writer

    content = state.content
    if content is None:
        file_path = state.file_path
        if not file_path.is_file():
            logger.error("文档切分跳过：缺少可切分的内容 path={}", file_path)
            writer(Step(message="正在切分文档", status=StepStatus.ERROR, error=f"Markdown 文件不存在：{file_path}"))
            return Command(goto=END)
        async with aiofiles.open(file_path, encoding="utf-8") as file:
            content = await file.read()
        content = content.replace("\r\n", "\n").replace("\r", "\n")

    writer(Step(message="正在切分文档"))
    chunks = []
    for title, line_no, body in split_by_header(content):
        # 章节正文再按 token 切小块，小块沿用所属章节的标题与标题行号
        for piece in md_split(body):
            chunks.append(Chunk(title=title, content=piece, header_index=line_no))

    logger.info("文档切分完成 chunks={}", len(chunks))
    writer(Step(message="正在切分文档", status=StepStatus.SUCCESS))
    return Command(update={"chunks": chunks}, goto="save_chunks")


embedder = Embedding()
milvus = Milvus()


async def save_chunks(state: State, runtime: Runtime[Context]) -> Command:
    """把 chunk 向量化后批量写入 Milvus 的 ``chunk`` 集合。

    正文取自 ``state.chunks``，连同运行时上下文的向量模型（``Context.embedding_model``）
    一次交给 :class:`~plugins.agents.embedding.Embedding` 生成稠密 + 稀疏向量（模型内部按
    :attr:`~plugins.agents.models.QwenEmbeddingModel.MAX_BATCH_SIZE` 自动分批），
    再连同 ``state.original_file_url`` / Markdown 文件名等元数据组装成
    :class:`~plugins.milvus.ChunkInserted` 记录，经
    :meth:`~plugins.milvus.Milvus.add_chunks` 写入；实体识别的节点已移除，chunk 记录的
    实体名字段落空值。

    Args:
        state: 流水线状态，使用其中的 ``chunks``、``original_file_url`` 与
            ``file_path``。
        runtime: LangGraph 运行时（携带 :class:`~plugins.agents.ingestion.schemas.Context`
            上下文，对话大模型等依赖按需从 ``runtime.context`` 取），步骤进度由
            ``stream_writer`` 上报。

    Returns:
        ``Command``：成功或无 chunk 时跳转尾节点（不修改状态）；实体名为空不算
        失败，按空值入库；写入失败时上报 ``error`` 并跳转尾节点。
    """
    writer = runtime.stream_writer

    chunks = state.chunks
    if not chunks:
        logger.info("chunk 入库跳过：没有可用的 chunk")
        writer(Step(message="正在写入向量库", status=StepStatus.SUCCESS))
        return Command(goto=END)

    file_name = state.file_path.name
    original_file_url = state.original_file_url

    logger.info(
        "chunk 入库开始 file={} chunks={}",
        file_name, len(chunks),
    )
    writer(Step(message="正在写入向量库"))

    try:
        dense, sparse = await embedder.embed(
            runtime.context.embedding_model, [chunk.content for chunk in chunks], "both"
        )
        if dense is None or sparse is None:
            logger.error("chunk 入库失败：文本向量返回为空 file={}", file_name)
            writer(Step(message="正在写入向量库", status=StepStatus.ERROR, error="文本向量返回为空"))
            return Command(goto=END)

        documents = [
            ChunkInserted(
                original_file_url=original_file_url,
                file_name=file_name,
                title=chunk.title,
                content=chunk.content,
                header_index=chunk.header_index,
                dense_vector=dense[index],
                sparse_vector=sparse[index],
            )
            for index, chunk in enumerate(chunks)
        ]
        await milvus.add_chunks(*documents)
    except Exception as exc:
        logger.exception("chunk 入库失败 file={}", file_name)
        writer(Step(message="正在写入向量库", status=StepStatus.ERROR, error=str(exc)))
        return Command(goto=END)

    logger.info("chunk 入库完成 file={} chunks={}", file_name, len(chunks))
    writer(Step(message="正在写入向量库", status=StepStatus.SUCCESS))
    return Command(goto=END)
