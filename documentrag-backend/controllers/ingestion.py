"""文档入库提交接口。

``POST /ingestion/documents``（**仅接受 PDF**）只做提交，不在请求里跑入库：

1. 校验文件名 / 扩展名 / 文件头魔数——失败返回普通 HTTP 400；
2. 生成文档 ID（UUID），按 :data:`CHUNK_SIZE` 分块把上传流落到
   ``config.STORAGE_DIR/<文档 ID>/<原文件名>``（失败返回 500 并清掉半成品文件）；
3. 以 :meth:`~plugins.snowflake.Snowflake.get_id` 生成任务 ID，建任务记录
   （:meth:`~plugins.redis.Task.create`：类型 ``document_ingest``、状态
   ``waiting``、输入信息与首条进度「PDF 已保存」），再交给调度器投递
   （:func:`~plugins.celery.tasks.submit`：有空位才真正执行，否则排队；
   投递失败返回 503）；
4. 立即返回任务记录字典（``Task.model_dump()``，状态可能是 ``waiting``）。

入库链路在 worker 进程里完成：读取内容提交 MinerU 解析 → 把 Markdown 与图片归档到
``storage/<文档 ID>/`` 并与原件一起上传 MinIO（对象元数据记文档 ID 与 Markdown 路径）
→ 入库智能体（图片描述、表格转写、切分、chunk 入库），每完成一步就把进度写回任务记录。
任务列表见 :mod:`controllers.tasks` 的 ``GET /tasks``，本地文件的列表 / 删除见
:mod:`controllers.documents`。
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile, status

import config
from logger import logger
from plugins.celery.tasks import TASK_TYPE, SAVED_MESSAGE, submit
from plugins.redis import Task, TaskStatus
from plugins.snowflake import Snowflake

router = APIRouter(prefix="/ingestion")

#: 允许的文档扩展名
PDF_SUFFIX = ".pdf"
#: PDF 文件头魔数：客户端声明的类型不可信，落盘时按内容校验真实格式
PDF_MAGIC = b"%PDF-"
#: 落盘分块大小（1 MiB），避免整份文件读进内存
CHUNK_SIZE = 1024 * 1024


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def submit_document(file: UploadFile = File(..., description="待入库的 PDF 文档")) -> dict:
    """接收 PDF 并投递入库任务，立即返回任务信息。

    Args:
        file: 表单字段 ``file`` 上传的 PDF（multipart/form-data）。

    Returns:
        任务记录字典（``Task.model_dump()``）：``id`` / ``type`` / ``status`` /
        ``error`` / ``inputs`` / ``outputs``（``inputs`` 是提交的输入，``outputs`` 的
        ``progress`` 键首条即「PDF 已保存」）。``status`` 为 ``waiting``（排队中，执行
        位置已满）或 ``running``（已投递）；后续进度从 ``GET /tasks`` 查询。

    Raises:
        HTTPException: 400——缺文件名、扩展名非 ``.pdf``、内容不是 PDF；500——
            PDF 落盘失败；503——任务记录写入失败（Redis 不可用）或投递失败
            （消息代理不可用）。
    """
    # 1. 文件名：只取纯文件名部分，丢弃客户端可能携带的路径
    filename = Path(file.filename or "").name.strip()
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="缺少文件名：请通过表单字段 file 上传 PDF 文档",
        )

    # 2. 扩展名必须是 .pdf
    if Path(filename).suffix.lower() != PDF_SUFFIX:
        logger.warning("提交文档被拒：非 PDF 文件 file={}", filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"仅支持 PDF 文档：{filename}",
        )

    # 3. 先读首块校验 %PDF- 魔数：此时尚未落盘，失败可直接返回 400
    head = await file.read(CHUNK_SIZE)
    if head[: len(PDF_MAGIC)] != PDF_MAGIC:
        logger.warning("提交文档被拒：文件内容不是 PDF file={} size={}B", filename, len(head))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="文件内容不是 PDF：请确认上传的是有效的 PDF 文档",
        )

    task_id = Snowflake.get_id()
    document_id = uuid4().hex
    root = Path(config.STORAGE_DIR)
    # 每份文档一个目录：storage/<文档 ID>/<原文件名>，解析产物随后归档到同一目录
    document_dir = root / document_id
    document_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = document_dir / filename
    # 对象键与本地目录同构，配合对象元数据里的文档 ID，两处能互相定位
    object_name = f"{document_id}/{filename}"

    # 4. 分块落盘（首块已在前面读取并校验），异常时删除半成品文件
    try:
        async with aiofiles.open(pdf_path, "wb") as out:
            await out.write(head)
            while chunk := await file.read(CHUNK_SIZE):
                await out.write(chunk)
    except Exception as exc:
        pdf_path.unlink(missing_ok=True)
        logger.exception(
            "提交文档失败：落盘异常 task_id={} document={} file={}",
            task_id, document_id, filename,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF 落盘失败：{exc}",
        ) from exc

    logger.info(
        "PDF 已落盘 task_id={} document={} path={} size={}B",
        task_id, document_id, pdf_path, pdf_path.stat().st_size,
    )

    # 5. 建任务记录（Redis），失败即 503：没有记录就无法查询进度
    try:
        task = await Task.create(
            task_id,
            task_type=TASK_TYPE,
            inputs={
                "document_id": document_id,
                "filename": filename,
                "pdf_path": str(pdf_path),
                "object_name": object_name,
            },
            message=SAVED_MESSAGE,
        )
    except Exception as exc:
        logger.exception("提交文档失败：任务记录写入异常 task_id={} file={}", task_id, filename)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"任务记录写入失败（Redis 不可用）：{exc}",
        ) from exc

    # 6. 交给调度器投递：有空位就立即执行，否则留在 waiting 排队；
    #    投递失败时把记录改成 error，避免留下永远等待的任务
    try:
        await submit(task)
    except Exception as exc:
        logger.exception("提交文档失败：任务投递异常 task_id={} file={}", task_id, filename)
        await task.finish(status=TaskStatus.ERROR, error=f"任务投递失败：{exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"任务投递失败（消息代理不可用）：{exc}",
        ) from exc

    logger.info("文档入库任务已提交 task_id={} file={} object={}", task_id, filename, object_name)
    # 投递时任务可能刚被置为 running（调度器读的是另一份记录），回读一次给出真实状态
    return ((await Task.fetch(task_id)) or task).model_dump()
