"""Celery 任务定义与调度：文档入库、数据集创建、数据集评测。

三类任务共用同一套任务记录（:class:`plugins.redis.Task`）与同一套排队规则：

- **排队**：任务由 API 进程提交时建为 ``waiting``，再由 :func:`submit` 触发投递；
  :func:`dispatch_waiting` 取出全部排队任务、按**创建时间先后**取最早的若干个，逐个置为
  ``running``（:meth:`~plugins.redis.Task.start`）后 ``delay`` 给 worker。「同时执行多少个」
  直接数 ``status=running`` 的任务记录，**所有类型合计同时至多
  :data:`~plugins.redis.TASK_MAX_CONCURRENT` 个**在执行，其余留在 ``waiting``。
- **接力**：任务跑完后由 :func:`_dispatch_next` 再拉一遍排队任务，因此一空出位置，
  下一个任务立即被投递（见 :func:`_execute`）。
- **进度**：每个任务把自己上报的步骤追加进任务记录的产出（``outputs.progress``），查询见
  :mod:`controllers.tasks` 的 ``GET /tasks``。

任务本身：

1. :func:`ingest_document`（``document_ingest``）：把一份已落盘的 PDF 跑完整条入库
   链路（读取内容 → MinerU 解析 → 归档到 ``storage/<文档 ID>/`` → 上传 MinIO → 入库
   智能体），输入为 ``document_id`` / ``filename`` / ``pdf_path`` / ``object_name``，
   产出原文 ``file_url``、Markdown 路径与正文；任一步失败都会清掉本次的解析产物
   （Markdown 与图片），只保留原件以便重试；
2. :func:`create_dataset`（``dataset_create``）：按文档 ID 取回文档的 Markdown 路径
   （对象元数据里的 ``markdown-path``），交给评估流水线读取、切分并生成测试用例；断点设在
   评测节点之前，所以生成节点落库后本次运行就停住，本任务只做到「数据集已落库」，输入为
   ``document_id`` / ``size`` / ``thread_id``，产出 ``dataset_id`` 与用例条数；
3. :func:`evaluate_dataset`（``dataset_eval``）：对已有数据集跑检索问答、指标打分与报告
   解读——一次 ``invoke`` 推进一轮（评测完跳回生成节点、又被断点拦下）；同一数据集可以反复
   提交评测（报告覆盖旧报告、不会重复生成数据集），输入为 ``dataset_id``，产出 ``result_id``
   与各指标均分。

任务体都是「同步入口 + 异步实现」：Celery 任务是同步函数，用
:func:`~plugins.celery.run_async` 在 worker 的常驻事件循环里跑异步实现；Redis 也用该
进程启动时（:func:`plugins.redis.connect`）建好的常驻连接。

用法::

    from plugins.celery.tasks import submit

    task = await Task.create(task_id, task_type="dataset_create", inputs={...}, ...)
    await submit(task)   # 有空位立即执行，否则排队（进度写回任务记录）
"""

import asyncio
import shutil
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

import aiofiles
import httpx

import config
from logger import logger
from models.dataset import Dataset, DatasetItem, Result
from plugins.agents.evaluation.graph import Evaluator
from plugins.agents.evaluation.schemas import Step, StepStatus
from plugins.agents.ingestion.graph import Ingestor
from plugins.agents.ingestion.schemas import StepStatus as IngestStepStatus
from plugins.celery import celery_client, run_async
from plugins.mineru import IMAGES_DIRNAME, Mineru, MineruAPIError, ParsedDocument
from plugins.minio import Minio, MinioError
from plugins.redis import Task, TaskStatus, TASK_MAX_CONCURRENT

__all__ = [
    "ACCEPTED_MESSAGE",
    "DATASET_TASK_TYPE",
    "EVALUATION_TASK_TYPE",
    "SAVED_MESSAGE",
    "TASK_TYPE",
    "TASK_TYPES",
    "create_dataset",
    "dispatch_waiting",
    "evaluate_dataset",
    "ingest_document",
    "submit",
]

#: 任务类型：文档入库
TASK_TYPE = "document_ingest"
#: 任务类型：数据集创建（生成测试用例）
DATASET_TASK_TYPE = "dataset_create"
#: 任务类型：数据集评测
EVALUATION_TASK_TYPE = "dataset_eval"

#: PDF 落盘（由提交接口完成）的进度说明
SAVED_MESSAGE = "PDF 已保存"
#: 上传原文到对象存储的进度说明
UPLOAD_MESSAGE = "正在上传原文"
#: MinerU 解析文档的进度说明
PARSE_MESSAGE = "正在解析文档"
#: 解析结果归档到 ``storage/<文档 ID>/`` 的进度说明
ARCHIVE_MESSAGE = "正在归档解析结果"
#: 数据集类任务的受理进度说明
ACCEPTED_MESSAGE = "任务已受理"

ingestor = Ingestor()
#: 评估智能体：进程内单例，任务体直接复用；``register`` 之后它就是带短期记忆的图
evaluator = Evaluator()
minio = Minio()
mineru = Mineru()

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# 调度：排队、占位、投递
# --------------------------------------------------------------------------- #
async def submit(task: Task) -> None:
    """投递任务：有空位就立即执行，否则留在 ``waiting`` 排队。

    排队由 :func:`dispatch_waiting` 统一处理：它数一遍执行中的任务、按创建时间先后把最早
    的若干个置为 ``running`` 再 ``delay``，因此同一时刻执行的任务不会超过
    :data:`~plugins.redis.TASK_MAX_CONCURRENT` 个。排队中的任务会在前一个任务收尾时被
    接力投递（见 :func:`_dispatch_next`）。

    Args:
        task: 已建好的任务记录（状态 ``waiting``）。
    """
    logger.info("任务已受理 task_id={} type={}", task.id, task.type)
    await dispatch_waiting()


async def dispatch_waiting() -> None:
    """投递排队中的任务：按创建时间取最早的若干个，置为 ``running`` 后 ``delay``。

    「同时执行多少个」直接数 ``status=running`` 的记录：本轮只补足到
    :data:`~plugins.redis.TASK_MAX_CONCURRENT` 个——没有执行中的任务时就是最早的五个，
    已有任务在执行时只补差额，因此排队再多也不会一次全拉起来。可以安全地并发 / 重复调用：
    :meth:`~plugins.redis.Task.start` 只在记录仍为 ``waiting`` 时才改状态，同一个任务不会
    被投递两次。没有排队任务或没有空位时直接返回；投递失败的任务置为 ``error``，避免它
    永远占着队首。
    """
    running = await Task.list(status=TaskStatus.RUNNING)
    capacity = TASK_MAX_CONCURRENT - len(running)
    if capacity <= 0:
        return
    for task in await Task.pop_earliest_waiting(capacity):
        if not await task.start():
            continue
        handler = TASK_TYPES.get(task.type)
        if handler is None:
            logger.error(
                "任务投递跳过：未知的任务类型 task_id={} type={}", task.id, task.type
            )
            await task.finish(
                status=TaskStatus.ERROR, error=f"未知的任务类型：{task.type}"
            )
            continue
        try:
            handler.delay(task.id)
        except Exception as exc:  # noqa: BLE001 - 投递失败也要把任务落到 error
            logger.opt(exception=exc).error(
                "任务投递失败 task_id={} type={}", task.id, task.type
            )
            await task.finish(status=TaskStatus.ERROR, error=f"任务投递失败：{exc}")
            continue
        logger.info("任务已投递 task_id={} type={}", task.id, task.type)


async def _dispatch_next() -> None:
    """接力投递排队中的任务（当前任务已收尾、空出一个执行位置）。

    投递失败只告警：当前任务的最终状态已经写好，不能因为接力失败改写它。
    """
    try:
        await dispatch_waiting()
    except Exception as exc:  # noqa: BLE001 - 接力失败不影响当前任务
        logger.opt(exception=exc).error("接力投递排队任务失败")


# --------------------------------------------------------------------------- #
# Celery 任务体：同步入口，统一外壳 _run_task
# --------------------------------------------------------------------------- #
@celery_client.task(name="ingestion.ingest_document")
def ingest_document(task_id: int) -> None:
    """后台执行一次文档入库（Celery 任务体，同步入口）。

    Args:
        task_id: 任务 ID；输入信息（PDF 路径、对象名）已由提交接口写在该任务
            记录里，进度与最终状态也写回同一条记录。
    """
    _run_task(task_id, _ingest)


@celery_client.task(name="evaluation.create_dataset")
def create_dataset(task_id: int) -> None:
    """后台生成一个数据集（Celery 任务体，同步入口）。

    Args:
        task_id: 任务 ID；输入（正文、用例条数、会话标识）已由提交接口写在该任务
            记录里，产出（数据集主键等）与最终状态也写回同一条记录。
    """
    _run_task(task_id, _create_dataset)


@celery_client.task(name="evaluation.evaluate_dataset")
def evaluate_dataset(task_id: int) -> None:
    """后台评测一个已有数据集（Celery 任务体，同步入口）。

    Args:
        task_id: 任务 ID；输入（数据集主键）已由提交接口写在该任务记录里，产出
            （报告主键、各指标均分）与最终状态也写回同一条记录。
    """
    _run_task(task_id, _evaluate_dataset)


#: 任务类型 → Celery 任务：调度时按任务记录的 ``type`` 分派
TASK_TYPES: dict[str, Any] = {
    TASK_TYPE: ingest_document,
    DATASET_TASK_TYPE: create_dataset,
    EVALUATION_TASK_TYPE: evaluate_dataset,
}


def _run_task(task_id: int, work: Callable[[Task], Awaitable[None]]) -> None:
    """Celery 任务体的统一外壳：驱动异步实现，异常照常抛出给 Celery 记录。

    Args:
        task_id: 任务 ID。
        work: 实际的异步实现（接收任务记录）。
    """
    try:
        run_async(_execute(task_id, work))
    except Exception:
        logger.exception("任务异常终止 task_id={}", task_id)
        raise


async def _execute(task_id: int, work: Callable[[Task], Awaitable[None]]) -> None:
    """读出任务记录、执行 ``work``，并在收尾时接力投递排队中的任务。

    兜底：``work`` 抛出且没来得及收尾时，把任务置为 ``error``——否则它会一直停在
    ``running``，白白占着一个执行位置。
    """
    task = await Task.fetch(task_id)
    if task is None:
        logger.error("任务跳过：任务记录不存在 task_id={}", task_id)
        await _dispatch_next()
        return
    try:
        await work(task)
    except Exception as exc:  # noqa: BLE001 - 任何异常都要落到任务状态上
        logger.opt(exception=exc).error("任务执行异常 task_id={}", task_id)
        await task.finish(status=TaskStatus.ERROR, error=f"任务执行失败：{exc}")
    finally:
        await _dispatch_next()


# --------------------------------------------------------------------------- #
# 文档入库
# --------------------------------------------------------------------------- #
async def _ingest(task: Task) -> None:
    """跑完整条入库链路，把每一步进度与产出写回任务记录。"""
    task_input = task.inputs
    document_id = task_input.get("document_id")
    filename = task_input.get("filename")
    object_name = task_input.get("object_name")
    pdf_path = Path(task_input.get("pdf_path") or "")
    if not document_id or not filename or not object_name:
        logger.error("入库任务跳过：任务记录缺少输入信息 task_id={}", task.id)
        await task.finish(status=TaskStatus.ERROR, error="任务记录缺少输入信息")
        return

    logger.info(
        "入库任务开始 task_id={} document={} pdf={}",
        task.id, document_id, pdf_path,
    )

    # 解析产物的落盘路径：归档成功后才有值，失败时据此清理（原件 PDF 始终保留）
    md_path: Path | None = None
    try:
        # 提交给 MinerU 之前先把内容读进内存：解析成功后才归档，解析失败的文档不会在
        # 存储里留下半成品（原件是提交接口落盘的，失败时由调用方决定是否清理）
        content = await asyncio.to_thread(pdf_path.read_bytes)
        parsed = await _run_stage(
            task,
            PARSE_MESSAGE,
            mineru.parse_content(content, filename, data_id=document_id),
        )
        md_path = await _run_stage(
            task,
            ARCHIVE_MESSAGE,
            _archive_document(document_id, filename, parsed),
        )
        file_url = await _run_stage(
            task,
            UPLOAD_MESSAGE,
            minio.upload_file(
                pdf_path,
                object_name,
                document_id=document_id,
                markdown_path=str(md_path),
            ),
        )
    except MinioError as exc:
        await _fail(task, f"PDF 上传对象存储失败：{exc}", md_path)
        return
    except MineruAPIError as exc:
        await _fail(task, f"文档解析失败：{exc}", md_path)
        return
    except TimeoutError as exc:
        await _fail(task, f"文档解析超时：{exc}", md_path)
        return
    except httpx.HTTPError as exc:
        await _fail(task, f"调用解析服务失败：{exc}", md_path)
        return
    except Exception as exc:  # noqa: BLE001 - 任何异常都要落到任务状态上
        logger.opt(exception=exc).error("入库任务失败 task_id={}", task.id)
        await _fail(task, f"入库任务执行失败：{exc}", md_path)
        return

    failed = await _run_agent(task, md_path, file_url)
    if failed:
        await _fail(task, failed, md_path)
        return

    try:
        async with aiofiles.open(md_path, encoding="utf-8") as md_file:
            content = await md_file.read()
    except Exception as exc:  # noqa: BLE001
        logger.opt(exception=exc).error(
            "入库任务失败：读取 Markdown task_id={} path={}", task.id, md_path
        )
        await _fail(task, f"读取 Markdown 失败：{exc}", md_path)
        return

    await task.finish(status=TaskStatus.SUCCESS, outputs={
        "document_id": document_id,
        "file_url": file_url,
        "file_path": str(md_path),
        "content": content,
    })
    logger.info(
        "入库任务完成 task_id={} document={} markdown={} chars={}",
        task.id, document_id, md_path, len(content),
    )


async def _archive_document(
        document_id: str, filename: str, parsed: ParsedDocument
) -> Path:
    """把解析结果归档到 ``storage/<文档 ID>/``，与原文案同目录。

    写入两份内容：

    - ``storage/<文档 ID>/<原文件名前缀>.md``：MinerU 的 ``full.md``；
    - ``storage/<文档 ID>/images/<文件名>``：随包图片（Markdown 里按 ``images/...``
      相对引用，必须与 Markdown 同级才能被解析到）。

    Args:
        document_id: 文档 ID（目录名，即提交接口生成的 UUID）。
        filename: 原文件名；Markdown 取它的前缀命名。
        parsed: MinerU 返回的 Markdown 与图片内容。

    Returns:
        落盘的 Markdown 路径。

    Raises:
        OSError: 目录创建或写文件失败。
    """
    document_dir = Path(config.STORAGE_DIR) / document_id
    md_path = document_dir / f"{Path(filename).stem}.md"

    def _write() -> None:
        document_dir.mkdir(parents=True, exist_ok=True)
        md_path.write_bytes(parsed.markdown)
        if not parsed.images:
            return
        images_dir = document_dir / IMAGES_DIRNAME
        images_dir.mkdir(parents=True, exist_ok=True)
        for name, data in parsed.images.items():
            (images_dir / name).write_bytes(data)

    # 写盘是阻塞 IO，丢到线程里，避免卡住 worker 的常驻事件循环
    await asyncio.to_thread(_write)
    logger.info(
        "入库文档已归档 document={} markdown={} images={}",
        document_id, md_path, len(parsed.images),
    )
    return md_path


async def _run_agent(task: Task, md_path: Path, file_url: str) -> str | None:
    """执行入库智能体，把各节点上报的步骤原样追加到任务进度。

    Args:
        task: 任务记录（进度直接写回它）。
        md_path: MinerU 解析产出的 Markdown 路径。
        file_url: 原文在 MinIO 中的地址，随 chunk 记录一并入库。

    Returns:
        流水线上报的失败原因（``None`` 表示全部步骤成功）；智能体本身抛出
        异常时返回该异常的说明。
    """
    failed: str | None = None
    try:
        async for step in ingestor.invoke(md_path, file_url):
            await task.add_step(
                step.message,
                TaskStatus(step.status.value),
                step.error,
            )
            if step.status is IngestStepStatus.ERROR:
                failed = step.error or f"{step.message}失败"
    except Exception as exc:  # noqa: BLE001 - 异常也要落到任务状态上
        logger.opt(exception=exc).error("入库智能体执行异常 task_id={}", task.id)
        return f"入库智能体执行失败：{exc}"
    return failed


# --------------------------------------------------------------------------- #
# 数据集创建 / 评测
# --------------------------------------------------------------------------- #
async def _create_dataset(task: Task) -> None:
    """按文档 ID 找到 Markdown 文档并生成测试用例，产出数据集主键与用例条数。

    输入只给 ``document_id``：文档正文由入库链路解析成 Markdown 并落盘，对象存储的元数据里
    记着它的路径（``markdown-path``），这里按文档 ID 取回路径后交给评估智能体去读、去切分。
    生成节点落库后跳到评测节点，断点在那里把它拦下，所以本任务做到「数据集已落库」就结束；
    用例的评测由 :func:`_evaluate_dataset` 单独发起（见 ``POST /evaluation/{dataset_id}``）。
    """
    task_input = task.inputs
    document_id = task_input.get("document_id")
    size = task_input.get("size")
    thread_id = task_input.get("thread_id")
    if not document_id or not isinstance(thread_id, int):
        logger.error("数据集创建任务跳过：任务记录缺少输入信息 task_id={}", task.id)
        await task.finish(status=TaskStatus.ERROR, error="任务记录缺少输入信息")
        return

    try:
        file = await minio.find_file(document_id)
    except MinioError as exc:
        await _fail(task, f"查找文档失败：{exc}")
        return
    if file is None:
        await _fail(task, f"文档不存在：{document_id}")
        return
    if not file.markdown_path:
        await _fail(task, f"文档缺少 Markdown 路径元数据：{document_id}")
        return

    logger.info(
        "数据集创建开始 task_id={} document={} markdown={} thread={} size={}",
        task.id, document_id, file.markdown_path, thread_id, size,
    )
    # 文档的读取与切分在评估流水线的 invoke 里完成：读不出来会以「正在读取文档」的失败
    # 步骤上报，因此这里只需要看流水线有没有上报失败
    failed = await _run_evaluator(
        task,
        evaluator.invoke(
            document_path=file.markdown_path, size=size, thread_id=thread_id
        ),
    )
    if failed:
        await _fail(task, failed)
        return

    # 数据集由流水线按会话标识落库，这里按 thread_id 取回主键（一个数据集一条线程）
    dataset = await Dataset.get_or_none(thread_id=thread_id)
    if dataset is None:
        await _fail(task, "数据集创建失败：没有找到生成的数据集")
        return

    items = await DatasetItem.filter(dataset_id=dataset.id).count()
    await task.finish(status=TaskStatus.SUCCESS, outputs={
        "dataset_id": dataset.id,
        "thread_id": thread_id,
        "name": dataset.name,
        "items": items,
    })
    logger.info(
        "数据集创建完成 task_id={} dataset={} items={}", task.id, dataset.id, items
    )


async def _evaluate_dataset(task: Task) -> None:
    """对已有数据集跑一轮评测，产出报告主键与各指标均分。

    一次 :meth:`~plugins.agents.evaluation.graph.Evaluator.invoke` 推进一轮：生成节点按
    ``dataset_id`` 只做路由，评测节点跑完检索问答、指标打分与报告落库后跳回生成节点，又被
    评测节点前的断点拦下，于是调用返回。想再评测就再调一次，每次覆盖旧报告，且不会重复生成
    数据集。会话标识取数据集上的 ``thread_id``，这样记忆可用
    :meth:`~plugins.agents.evaluation.graph.Evaluator.forget` 一并删除。
    """
    task_input = task.inputs
    dataset_id = task_input.get("dataset_id")
    if not isinstance(dataset_id, int):
        logger.error("数据集评测任务跳过：任务记录缺少输入信息 task_id={}", task.id)
        await task.finish(status=TaskStatus.ERROR, error="任务记录缺少输入信息")
        return

    dataset = await Dataset.get_or_none(id=dataset_id)
    if dataset is None:
        await _fail(task, f"数据集不存在：{dataset_id}")
        return

    thread_id = dataset.thread_id
    logger.info(
        "数据集评测开始 task_id={} dataset={} thread={}", task.id, dataset_id, thread_id
    )
    failed = await _run_evaluator(
        task, evaluator.invoke(dataset_id=dataset_id, thread_id=thread_id)
    )
    if failed:
        await _fail(task, failed)
        return

    # 报告由评测节点落库（同一数据集重新评测会覆盖旧报告），这里取回最新的一份
    result = await Result.filter(dataset_id=dataset_id).order_by("-id").first()
    await task.finish(status=TaskStatus.SUCCESS, outputs={
        "dataset_id": dataset_id,
        "thread_id": thread_id,
        "result_id": result.id if result is not None else None,
        "summary": result.summary if result is not None else {},
    })
    logger.info(
        "数据集评测完成 task_id={} dataset={} result={}",
        task.id, dataset_id, result.id if result is not None else None,
    )


async def _run_evaluator(task: Task, steps: AsyncIterator[Step]) -> str | None:
    """把评估流水线上报的步骤原样追加到任务进度。

    Args:
        task: 任务记录（进度直接写回它）。
        steps: 评估流水线的步骤流（``Evaluator.invoke``）。

    Returns:
        流水线上报的失败原因（``None`` 表示全部步骤成功）。
    """
    failed: str | None = None
    async for step in steps:
        await task.add_step(step.message, TaskStatus(step.status.value), step.error)
        if step.status is StepStatus.ERROR:
            failed = step.error or f"{step.message}失败"
    return failed


# --------------------------------------------------------------------------- #
# 公共小工具
# --------------------------------------------------------------------------- #
async def _run_stage(task: Task, message: str, work: Awaitable[T]) -> T:
    """执行一个阶段并上报进度：``running`` → 执行 → ``success`` / ``error``。

    Args:
        task: 任务记录（进度直接写回它）。
        message: 阶段说明，用同一条文案上报该阶段的开始与结束。
        work: 待执行的协程（已构造，避免上报 ``running`` 之前就开始干活）。

    Returns:
        协程的返回值。

    Raises:
        Exception: 阶段失败时原样抛出（失败已记录到进度里）。
    """
    await task.add_step(message)
    try:
        result = await work
    except Exception as exc:
        await task.add_step(message, TaskStatus.ERROR, str(exc))
        raise
    await task.add_step(message, TaskStatus.SUCCESS)
    return result


async def _fail(
        task: Task, detail: str, markdown_path: Path | None = None
) -> None:
    """标记任务失败：记录原因、置为 ``error``，并清掉本次的解析产物。

    Args:
        task: 任务记录。
        detail: 失败原因，写入 ``error`` 字段。
        markdown_path: 本次归档出的 Markdown 路径；给了就一并删除 Markdown 与图片目录，
            原件（PDF）保留——重试要靠它。
    """
    logger.error("任务失败 task_id={} err={}", task.id, detail)
    await task.finish(status=TaskStatus.ERROR, error=detail)
    if markdown_path is not None:
        await _discard_archive(markdown_path)


async def _discard_archive(md_path: Path) -> None:
    """删除失败任务留下的解析产物：Markdown 与其同级的 ``images/`` 目录。

    解析产物是失败任务用不上的中间结果，重试时会重新生成，留着只会占空间；**原件 PDF
    不删**——它在同一目录下，是重试的唯一依据。清理失败只告警，不改写任务状态。

    Args:
        md_path: :func:`_archive_document` 落盘的 Markdown 路径。
    """

    def _remove() -> None:
        md_path.unlink(missing_ok=True)
        images_dir = md_path.parent / IMAGES_DIRNAME
        if images_dir.is_dir():
            shutil.rmtree(images_dir, ignore_errors=True)

    try:
        await asyncio.to_thread(_remove)
    except OSError as exc:  # noqa: BLE001 - 清理失败不影响任务最终状态
        logger.opt(exception=exc).warning("清理解析产物失败 markdown={}", md_path)
        return
    logger.info("已清理解析产物 markdown={}", md_path)
