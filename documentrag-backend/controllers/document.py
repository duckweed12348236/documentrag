"""文件接口：列出 / 删除 MinIO 对象。

两个接口都直接复用 :class:`~plugins.minio.Minio` 的方法，不在控制器里碰对象存储的细节：

- ``GET /document?prefix=``：:meth:`~plugins.minio.Minio.list_files` 列出对象，返回对象键、
  文档 ID、大小、内容类型、修改时间、ETag 与直链——Markdown 路径与完整元数据属于内部实现
  细节，不对外返回；
- ``DELETE /document/{object_name}``：:meth:`~plugins.minio.Minio.delete_file` 删除对象，
  幂等（对象不存在同样成功）。

对象键形如 ``<文档 ID>/<原文件名>``（与本地 ``storage/<文档 ID>/`` 同构），带斜杠，因此
删除接口用 ``{object_name:path}`` 接住整段路径；中文、空格按 URL 编码传入即可。

对象存储不可用时两个接口都返回 502——与项目其它插件一致，把上游故障如实暴露给调用方，
不伪装成 404 / 204。
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Query, status

from logger import logger
from plugins.minio import Minio, MinioError

router = APIRouter(prefix="/document")
minio = Minio()

#: 列表接口不返回的字段：Markdown 路径与完整元数据是内部实现细节，调用方用不到
EXCLUDED_FIELDS = ("markdown_path", "metadata")


@router.get("")
async def list_files(
        prefix: str = Query(
            "", description="只列出键以此开头的对象，如 ``6f1c...e9/``；缺省列出全部"
        ),
) -> list[dict]:
    """列出存储桶中的文件。

    只返回调用方需要的字段：Markdown 路径（``markdown_path``）与完整元数据
    （``metadata``）属于内部实现细节，由 :data:`EXCLUDED_FIELDS` 过滤掉。

    Args:
        prefix: 对象键前缀；缺省列出全部对象。

    Returns:
        对象字典列表（``ObjectInfo`` 逐字段展开后过滤）：``object_name`` /
        ``document_id`` / ``size`` / ``content_type`` / ``last_modified`` / ``etag`` /
        ``url``。

    Raises:
        HTTPException: 502——对象存储不可用或读取元数据失败。
    """
    try:
        files = await minio.list_files(prefix)
    except MinioError as exc:
        logger.opt(exception=exc).error("列出文件失败 prefix={}", prefix)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"对象存储不可用：{exc}",
        ) from exc
    return [
        {
            key: value
            for key, value in asdict(item).items()
            if key not in EXCLUDED_FIELDS
        }
        for item in files
    ]


@router.delete("/{object_name:path}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(object_name: str) -> None:
    """删除对象（幂等：对象不存在同样返回 204）。

    Args:
        object_name: 对象键，如 ``6f1c...e9/spec.pdf``；带斜杠，按整段路径解析。

    Raises:
        HTTPException: 400——对象键为空；502——对象存储不可用或删除失败。
    """
    try:
        await minio.delete_file(object_name)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except MinioError as exc:
        logger.opt(exception=exc).error("删除文件失败 object={}", object_name)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"对象存储不可用：{exc}",
        ) from exc
    logger.info("文件已删除 object={}", object_name)
