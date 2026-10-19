"""MinIO 对象存储客户端插件（文件上传）。

MinIO 官方 SDK ``minio-py`` 只提供**同步** API，本插件用
``asyncio.to_thread`` 把阻塞调用卸载到线程池，避免阻塞事件循环，对外
仍暴露统一的 ``async`` 接口。

启动流程（``register``）：

1. 依据 ``config.MINIO_*`` 构造客户端（构造过程无网络 IO）；
2. 确保默认存储桶 ``config.MINIO_BUCKET`` 存在，不存在则创建；
3. 若 ``config.MINIO_PUBLIC_READ`` 为真，把存储桶设为**匿名只读**
   （公开读，等价 ``mc anonymous set download``），使 :meth:`Minio.build_object_url`
   给出的直链无需签名即可被外部（浏览器 / 第三方系统）直接访问。

2 / 3 两步失败都只告警不抛错（fail-open，与项目其它插件一致）：存储桶
初始化失败延迟到首次上传重试；公开读策略设置失败时上传等功能不受影响，
但对象 URL 匿名访问会返回 ``AccessDenied``。

使用::

    from plugins.minio import Minio

    client = Minio()                              # 进程内单例
    await client.register()
    key = "6f1c...e9/spec.pdf"                    # 调用方自行生成 key
    url = await client.upload_file(
        "storage/6f1c...e9/spec.pdf",
        key,
        document_id="6f1c...e9",                  # 文档 ID（UUID）
        markdown_path="storage/6f1c...e9/spec.md",
    )
    files = await client.list_files("6f1c...e9/")   # 大小 / 类型 / 修改时间 / 元数据
    await client.delete_file(key)
    await client.close()
    print(url)  # → http://localhost:9000/productrag/6f1c...e9/spec.pdf

上传时会一并写入**对象元数据**（S3 自动加 ``X-Amz-Meta-`` 前缀）：

- :data:`DOCUMENT_ID_META_KEY`（``id``）：文档 ID（UUID），与本地 ``storage/<id>/``
  目录一一对应；
- :data:`MARKDOWN_PATH_META_KEY`（``markdown-path``）：该文件（PDF）转换出的 Markdown
  文档路径，原件与解析产物靠它对应起来。

两者都可缺省，都不给时不写自定义元数据。元数据的值会做**百分号编码**：S3 只接受
US-ASCII 的元数据值（对象键不受此限），而 Markdown 路径里常带中文原文件名；读取时
:meth:`list_files` 会 ``unquote`` 还原成原样。
上传接口返回的仍是对象的**可访问 URL**（path-style，
``<scheme>://<endpoint>/<bucket>/<object>``），不再只返回对象名；对象名
本身已在 URL 中，且调用方本就持有。基础域名取
``config.MINIO_PUBLIC_ENDPOINT``，留空时回退 ``config.MINIO_ENDPOINT``，
协议由 ``config.MINIO_SECURE`` 决定（端点自带协议前缀时以其为准）。

对象命名由调用方掌握：``upload_file`` / ``upload_files`` 都不自动构造对象键，
要求 ``object_name`` 必传。如需按 ``YYYY/MM/DD/<主名>_<雪花ID><扩展名>`` 生成唯一键，
请在调用方自行拼接——按日期分目录便于在控制台按时间浏览，雪花 ID 保证同名文件不会
互相覆盖。

除此之外还提供 :meth:`Minio.find_file`（按文档 ID 查找对象，取它的 Markdown 路径）、
:meth:`Minio.list_files`（列出对象及其元数据）与 :meth:`Minio.delete_file`（删除对象）：
查找与列出都要对每个对象各取一次 ``stat_object``，因此既有大小 / 类型 / 修改时间，也有
完整的自定义元数据；删除是 S3 语义的**幂等**操作，对象不存在同样返回成功。

日志：通过 :mod:`logger` 暴露的 loguru 实例输出——

- ``info`` ：客户端就绪、存储桶创建、公开读策略生效、上传开始/完成、列出/删除对象、客户端关闭
- ``warning``：启动阶段存储桶初始化失败（延迟到首次上传重试）、公开读策略设置失败
- ``error``：存储桶初始化、策略设置、上传、列出或删除失败（含 S3 错误码与消息）
"""

from __future__ import annotations

import asyncio
import mimetypes
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from minio import Minio as MinioClient
from minio.error import S3Error

import config
from logger import logger
from plugins import Plugin
from plugins.json import dump as json_dump

__all__ = ["Minio", "MinioError", "ObjectInfo"]

#: 无法从扩展名推断 MIME 时的兜底类型
DEFAULT_CONTENT_TYPE = "application/octet-stream"
#: 对象元数据里保存文档 ID（UUID）的键：S3 会自动加 ``X-Amz-Meta-`` 前缀
DOCUMENT_ID_META_KEY = "id"
#: 对象元数据里保存 Markdown 文档路径的键：S3 会自动加 ``X-Amz-Meta-`` 前缀
MARKDOWN_PATH_META_KEY = "markdown-path"
#: 默认对象名前缀的日期格式（按天分目录）
DATE_PREFIX_FORMAT = "%Y/%m/%d"
#: 桶策略文档版本（AWS IAM policy 语法，MinIO 沿用）
POLICY_VERSION = "2012-10-17"

#: 批量上传的项目：``(本地路径, 对象键)`` 或
#: ``(本地路径, 对象键, 文档 ID, Markdown 路径)``（后两项可为 ``None``）
UploadItem = (
    tuple[str | Path, str]
    | tuple[str | Path, str, str | None, str | None]
)


class MinioError(RuntimeError):
    """MinIO 操作失败：对 ``minio.error.S3Error`` 的统一封装。"""

    def __init__(self, message: str, *, code: str = "") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ObjectInfo:
    """存储桶里的一个对象：标识、大小、类型、修改时间与自定义元数据。"""

    #: 对象键（bucket 内的完整路径）
    object_name: str
    #: 文档 ID（UUID）：取自对象元数据（见 :data:`DOCUMENT_ID_META_KEY`），未写入时为 ``None``
    document_id: str | None
    #: 该文件（PDF）转换出的 Markdown 文档路径：取自对象元数据
    #: （见 :data:`MARKDOWN_PATH_META_KEY`），未写入时为 ``None``
    markdown_path: str | None
    #: 对象大小（字节）
    size: int
    #: 内容类型（MIME），如 ``application/pdf``
    content_type: str
    #: 最后修改时间（UTC）
    last_modified: datetime | None
    #: 实体标签（去掉两侧引号）
    etag: str
    #: 可直接访问的对象 URL（见 :meth:`Minio.build_object_url`）
    url: str
    #: 对象的全部元数据（``stat_object`` 原样返回，含 ``X-Amz-Meta-*`` 与内容类型等）
    metadata: dict[str, str]


class Minio(Plugin):
    """MinIO 客户端：负责对象上传与存储桶自动初始化。

    ``Plugin`` 的元类已保证进程内单例，直接 ``Minio()`` 取同一实例即可。
    """

    def __init__(self) -> None:
        self._client: MinioClient | None = None
        self._bucket: str = config.MINIO_BUCKET
        self._bucket_ready = False
        # 拼接对外 URL 用的基础地址；未单独配置时回退服务端点
        self._public_endpoint: str = (
            config.MINIO_PUBLIC_ENDPOINT or config.MINIO_ENDPOINT
        )

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    async def register(self) -> None:
        """构造客户端并确保存储桶存在。

        存储桶初始化失败**不阻断** API 启动（fail-open）：MinIO 短暂不可用
        或配置有误时应用仍能起来，首次上传会再次尝试创建并暴露真实错误。
        """
        self._client = MinioClient(
            endpoint=config.MINIO_ENDPOINT,
            access_key=config.MINIO_ACCESS_KEY or None,
            secret_key=config.MINIO_SECRET_KEY or None,
            secure=config.MINIO_SECURE,
        )
        logger.info(
            "MinIO 客户端已就绪 endpoint={} bucket={} secure={}",
            config.MINIO_ENDPOINT,
            self._bucket,
            config.MINIO_SECURE,
        )
        try:
            await self.ensure_bucket()
        except Exception:
            logger.warning(
                "MinIO 存储桶初始化失败 bucket={}，将在首次上传时重试",
                self._bucket,
            )

    async def close(self) -> None:
        """释放客户端引用（幂等）。

        minio-py 内部基于 urllib3 连接池，没有显式 ``close``；置空引用后由
        GC 回收，事件循环不再持有连接。
        """
        self._client = None
        self._bucket_ready = False
        logger.info("MinIO 客户端已关闭")

    @property
    def client(self) -> MinioClient:
        """惰性取客户端；未注册时抛错，避免在请求处理里埋雷。"""
        if self._client is None:
            raise RuntimeError(
                "MinIO 客户端未注册：请确认 plugins 已随 FastAPI lifespan 启动，"
                "或先显式调用 Minio().register()"
            )
        return self._client

    # ------------------------------------------------------------------ #
    # 存储桶
    # ------------------------------------------------------------------ #
    async def ensure_bucket(self) -> None:
        """确保默认存储桶存在并按配置放开外部访问；幂等，已就绪时立即返回。

        Raises:
            MinioError: 查询或创建存储桶时发生 S3 错误。
        """
        if self._bucket_ready:
            return

        client = self.client
        try:
            exists = await asyncio.to_thread(client.bucket_exists, self._bucket)
            if not exists:
                await asyncio.to_thread(client.make_bucket, self._bucket)
                logger.info("MinIO 存储桶已创建 bucket={}", self._bucket)
        except S3Error as exc:
            logger.error(
                "MinIO 存储桶初始化失败 bucket={} code={} err={}",
                self._bucket,
                exc.code,
                exc.message,
            )
            raise MinioError(
                f"MinIO 存储桶初始化失败 bucket={self._bucket}: "
                f"{str(exc.code)} {str(exc.message)}",
                code=str(exc.code),
            ) from exc

        if config.MINIO_PUBLIC_READ:
            try:
                await self.set_public_read()
            except MinioError:
                # 公开读失败不阻断上传（fail-open），仅提示外部直链不可用
                logger.warning(
                    "MinIO 存储桶未设为匿名只读 bucket={}，对象 URL 匿名访问将返回 "
                    "AccessDenied；可在控制台或执行 "
                    "mc anonymous set download <alias>/{} 后重试",
                    self._bucket,
                    self._bucket,
                )

        self._bucket_ready = True

    async def set_public_read(self) -> None:
        """把默认存储桶设为匿名只读（公开读），供外部直接访问对象 URL。

        等价于控制台把桶的 ``Access Rules`` 设为 ``public``，或执行
        ``mc anonymous set download <alias>/<bucket>``：策略只放行
        ``s3:GetObject``，**不放行** 列举（``ListBucket``）与任何写操作，
        因此外部可读取已知对象名（即 :meth:`build_object_url` 给出的直链），
        但无法遍历桶内容。幂等：重复调用只是覆盖同一份策略。

        Raises:
            MinioError: 设置策略失败（权限不足、端点不可达等）。
        """
        policy = json_dump({
            "Version": POLICY_VERSION,
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{self._bucket}/*"],
                }
            ],
        })
        try:
            await asyncio.to_thread(
                self.client.set_bucket_policy, self._bucket, policy
            )
        except S3Error as exc:
            logger.error(
                "MinIO 公开读策略设置失败 bucket={} code={} err={}",
                self._bucket,
                exc.code,
                exc.message,
            )
            raise MinioError(
                f"MinIO 公开读策略设置失败 bucket={self._bucket}: "
                f"{str(exc.code)} {str(exc.message)}",
                code=str(exc.code),
            ) from exc

        logger.info("MinIO 存储桶已设为匿名只读（外部可直接访问）bucket={}", self._bucket)

    # ------------------------------------------------------------------ #
    # 访问地址
    # ------------------------------------------------------------------ #
    def build_object_url(self, object_name: str) -> str:
        """按 path-style 拼接对象的可访问 URL。

        ``<scheme>://<endpoint>/<bucket>/<object>``：

        - 端点取 ``config.MINIO_PUBLIC_ENDPOINT``，留空时回退
          ``config.MINIO_ENDPOINT``；端点自带 ``http(s)://`` 前缀时以它为准，
          否则由 ``config.MINIO_SECURE`` 决定协议。
        - 对象名中的 ``/`` 保留为路径分隔，其余字符（中文、空格等）做百分号
          编码。

        Args:
            object_name: bucket 内的对象键。

        Returns:
            可直接访问的对象 URL。

        Raises:
            ValueError: ``object_name`` 为空。
        """
        if not object_name:
            raise ValueError("build_object_url 必须显式提供 object_name")

        endpoint = self._public_endpoint.strip().rstrip("/")
        if "://" in endpoint:
            base = endpoint
        else:
            scheme = "https" if config.MINIO_SECURE else "http"
            base = f"{scheme}://{endpoint}"
        return f"{base}/{self._bucket}/{quote(object_name.lstrip('/'), safe='/')}"

    # ------------------------------------------------------------------ #
    # 上传
    # ------------------------------------------------------------------ #
    async def upload_file(
            self,
            file_path: str | Path,
            object_name: str,
            *,
            document_id: str | None = None,
            markdown_path: str | None = None,
    ) -> str:
        """上传本地文件到 MinIO（可一并写入文档 ID 与 Markdown 路径元数据）。

        ``object_name`` **必须**由调用方显式提供——本函数不自动构造对象键，把命名权
        交还调用方。``document_id`` / ``markdown_path`` 给了就写入对象元数据
        （见 :data:`DOCUMENT_ID_META_KEY` / :data:`MARKDOWN_PATH_META_KEY`），把对象与
        本地文档目录、解析产物对应起来，:meth:`list_files` 会一并返回；都不给则不写
        自定义元数据。

        Args:
            file_path: 本地文件路径；不存在时抛 ``FileNotFoundError``。
            object_name: 对象存储中的键，**必填**，空字符串会抛
                :class:`ValueError`。
            document_id: 文档 ID（UUID），写入对象元数据；缺省不写。
            markdown_path: 该文件（PDF）转换出的 Markdown 文档路径，写入对象元数据；
                缺省不写。MIME 类型按扩展名推断，推断不出时用
                ``application/octet-stream``。

        Returns:
            上传后对象的可访问 URL（path-style，见
            :meth:`build_object_url`）。

        Raises:
            ValueError: ``object_name`` 为空。
            FileNotFoundError: ``file_path`` 不存在。
            MinioError: 存储桶不可用，或上传过程中发生 S3 错误。
        """
        if not object_name:
            raise ValueError("upload_file 必须显式提供 object_name")

        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")

        key = object_name
        content_type = self.guess_content_type(path.name)
        metadata = self._build_metadata(document_id, markdown_path)

        await self.ensure_bucket()
        logger.info(
            "MinIO 开始上传 bucket={} object={} file={} size={}B content_type={} "
            "document_id={} markdown_path={}",
            self._bucket,
            key,
            path,
            path.stat().st_size,
            content_type,
            document_id,
            markdown_path,
        )
        try:
            await asyncio.to_thread(
                self.client.fput_object,
                self._bucket,
                key,
                str(path),
                content_type=content_type,
                metadata=metadata,
            )
        except S3Error as exc:
            logger.error(
                "MinIO 上传失败 bucket={} object={} code={} err={}",
                self._bucket,
                key,
                exc.code,
                exc.message,
            )
            raise MinioError(
                f"MinIO 上传失败 object={key}: {str(exc.code)} {str(exc.message)}",
                code=str(exc.code),
            ) from exc

        url = self.build_object_url(key)
        logger.info(
            "MinIO 上传完成 bucket={} object={} url={}", self._bucket, key, url
        )
        return url

    @staticmethod
    def _build_metadata(
            document_id: str | None, markdown_path: str | None
    ) -> dict[str, str] | None:
        """拼出自定义元数据；两项都没给时返回 ``None``（不写自定义元数据）。

        S3 的自定义元数据值必须是 **US-ASCII**（对象键不受限，元数据受限），而 Markdown
        路径里几乎必然含中文原文件名，直接放进去会被 minio 的 ``to_string`` 拒绝、抛出
        ``unsupported metadata value``。所以这里统一做百分号编码，读取时由
        :meth:`_meta_value` 还原。
        """
        metadata = {}
        if document_id:
            metadata[DOCUMENT_ID_META_KEY] = quote(document_id, safe="")
        if markdown_path:
            # 路径分隔符保留原样，读起来仍是一条路径；中文/空格等编码成 %XX
            metadata[MARKDOWN_PATH_META_KEY] = quote(markdown_path, safe="/")
        return metadata or None

    @staticmethod
    def _resolve_upload_item(item: UploadItem) -> tuple[Path, str, str | None, str | None]:
        """校验批量上传的单条项目，返回 ``(Path, 对象键, 文档 ID, Markdown 路径)``。

        Raises:
            ValueError: ``object_name`` 为空，或项目不是二元 / 四元组。
            FileNotFoundError: ``file_path`` 不存在。
        """
        if len(item) == 2:
            file_path, object_name = item  # type: ignore[misc]
            document_id = markdown_path = None
        elif len(item) == 4:
            file_path, object_name, document_id, markdown_path = item  # type: ignore[misc]
        else:
            raise ValueError(
                "批量上传的项目应为 (file_path, object_name) 或 "
                f"(file_path, object_name, document_id, markdown_path)，收到 {item!r}"
            )
        if not object_name:
            raise ValueError(
                "批量上传必须显式提供 object_name，不接受空字符串"
            )
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")
        return path, object_name, document_id or None, markdown_path or None

    async def upload_files(
            self,
            items: Sequence[UploadItem],
    ) -> list[str]:
        """批量上传本地文件到 MinIO（逐个写入文档 ID / Markdown 路径元数据）。

        ``object_name`` **必须**由调用方显式提供——本函数不自动构造对象键，避免批量
        场景下命名策略与调用方预期不一致。

        Args:
            items: 有序序列，每项是 ``(file_path, object_name)`` 或
                ``(file_path, object_name, document_id, markdown_path)``：
                - ``file_path``：本地文件路径。
                - ``object_name``：对象存储中的键，**必填**，空字符串会抛
                  :class:`ValueError`。
                - ``document_id``：可选，文档 ID（UUID），写入对象元数据
                  （见 :data:`DOCUMENT_ID_META_KEY`）。
                - ``markdown_path``：可选，该文件转换出的 Markdown 文档路径，写入对象
                  元数据（见 :data:`MARKDOWN_PATH_META_KEY`）。

        Returns:
            与 ``items`` 等长的对象 URL 列表（path-style，见
            :meth:`build_object_url`），顺序与 ``items`` 严格一致。

        Raises:
            ValueError: 任一 ``object_name`` 为空，或项目结构不对。
            FileNotFoundError: 任一 ``file_path`` 不存在（前置校验，避免半
                上传后才暴露缺文件）。
            MinioError: 存储桶不可用，或任一文件上传过程中发生 S3 错误
                （首次失败即抛出，未上传的文件不会继续）。
        """
        if not items:
            return []

        # 前置校验：缺文件 / 空对象名立即抛出，避免部分上传后才暴露问题
        validated = [self._resolve_upload_item(item) for item in items]

        await self.ensure_bucket()

        total = len(validated)
        results = []
        for index, (path, key, document_id, markdown_path) in enumerate(validated, start=1):
            content_type = self.guess_content_type(path.name)
            metadata = self._build_metadata(document_id, markdown_path)
            logger.info(
                "MinIO 开始批量上传 [{}/{}] bucket={} object={} file={} "
                "size={}B content_type={} document_id={} markdown_path={}",
                index,
                total,
                self._bucket,
                key,
                path,
                path.stat().st_size,
                content_type,
                document_id,
                markdown_path,
            )
            try:
                await asyncio.to_thread(
                    self.client.fput_object,
                    self._bucket,
                    key,
                    str(path),
                    content_type=content_type,
                    metadata=metadata,
                )
            except S3Error as exc:
                logger.error(
                    "MinIO 批量上传失败 bucket={} object={} code={} err={} "
                    "done={}/{}",
                    self._bucket,
                    key,
                    exc.code,
                    exc.message,
                    index - 1,
                    total,
                )
                raise MinioError(
                    f"MinIO 批量上传失败 object={key}: {str(exc.code)} {str(exc.message)}",
                    code=str(exc.code),
                ) from exc

            url = self.build_object_url(key)
            logger.info(
                "MinIO 批量上传完成 [{}/{}] bucket={} object={} url={}",
                index,
                total,
                self._bucket,
                key,
                url,
            )
            results.append(url)

        logger.info(
            "MinIO 批量上传全部完成 total={} bucket={}", total, self._bucket
        )
        return results

    # ------------------------------------------------------------------ #
    # 列出 / 删除
    # ------------------------------------------------------------------ #
    async def list_files(self, prefix: str = "") -> list[ObjectInfo]:
        """列出存储桶中的对象（含原文件名等元数据）。

        递归列出全部对象，再对每个对象取一次 ``stat_object``：列表接口本身只带大小 /
        修改时间 / ETag，内容类型与自定义元数据（原文件名）要靠 stat 取回，因此请求数是
        「1 次列表 + N 次 stat」——对象很多时请配 ``prefix`` 缩小范围。

        Args:
            prefix: 只列出键以此开头的对象，如 ``2026/09/22/``；缺省列出全部。

        Returns:
            对象信息列表（:class:`ObjectInfo`），顺序与存储服务返回一致（按键名字典
            序）；桶为空时为空列表。

        Raises:
            MinioError: 存储桶不可用，或列出 / 读取元数据过程中发生 S3 错误。
        """
        await self.ensure_bucket()
        try:
            return await asyncio.to_thread(self._collect_files, prefix)
        except S3Error as exc:
            logger.error(
                "MinIO 列出对象失败 bucket={} prefix={} code={} err={}",
                self._bucket,
                prefix,
                exc.code,
                exc.message,
            )
            raise MinioError(
                f"MinIO 列出对象失败 prefix={prefix}: "
                f"{str(exc.code)} {str(exc.message)}",
                code=str(exc.code),
            ) from exc

    def _collect_files(self, prefix: str) -> list[ObjectInfo]:
        """同步收集对象信息（在 ``to_thread`` 里执行，避免阻塞事件循环）。"""
        files = [
            self._build_object_info(obj)
            for obj in self.client.list_objects(self._bucket, prefix=prefix, recursive=True)
        ]
        logger.info(
            "MinIO 列出对象完成 bucket={} prefix={} files={}",
            self._bucket,
            prefix,
            len(files),
        )
        return files

    def _build_object_info(self, obj: Any) -> ObjectInfo:
        """把一个列表项补上 ``stat_object`` 的元数据，组装成 :class:`ObjectInfo`。

        ``list_objects`` 只给键、大小、修改时间与 ETag；内容类型和自定义元数据（文档 ID、
        Markdown 路径）要单独 ``stat`` 一次才能拿到。

        Args:
            obj: ``list_objects`` 产出的对象信息。

        Returns:
            组装好的 :class:`ObjectInfo`（``url`` 用 :meth:`build_object_url` 拼出）。
        """
        stat = self.client.stat_object(self._bucket, obj.object_name)
        metadata = dict(stat.metadata or {})
        return ObjectInfo(
            object_name=obj.object_name,
            document_id=self._get_meta_value(metadata, DOCUMENT_ID_META_KEY),
            markdown_path=self._get_meta_value(metadata, MARKDOWN_PATH_META_KEY),
            size=stat.size or 0,
            content_type=stat.content_type or self.guess_content_type(obj.object_name),
            last_modified=stat.last_modified or obj.last_modified,
            etag=(obj.etag or "").strip('"'),
            url=self.build_object_url(obj.object_name),
            metadata=metadata,
        )

    def _find_file(self, document_id: str) -> ObjectInfo | None:
        """按文档 ID 同步查找对象（在 ``to_thread`` 里执行）。"""
        prefix = f"{document_id}/"
        for obj in self.client.list_objects(self._bucket, prefix=prefix, recursive=True):
            info = self._build_object_info(obj)
            if info.document_id == document_id:
                return info
        return None

    async def find_file(self, document_id: str) -> ObjectInfo | None:
        """按文档 ID 查找对象（找不到返回 ``None``）。

        先按对象键前缀 ``<文档 ID>/`` 缩小扫描范围（上传时对象键就是这个规则），再逐个
        ``stat_object`` 比对元数据里的 ``id``——最终以元数据为准，前缀只用来减少请求。

        Args:
            document_id: 文档 ID（提交入库时生成的 UUID）。

        Returns:
            命中的 :class:`ObjectInfo`（含 ``markdown_path``）；没有匹配对象时为 ``None``。

        Raises:
            ValueError: ``document_id`` 为空。
            MinioError: 存储桶不可用，或读取元数据过程中发生 S3 错误。
        """
        if not document_id:
            raise ValueError("find_file 必须显式提供 document_id")

        await self.ensure_bucket()
        try:
            info = await asyncio.to_thread(self._find_file, document_id)
        except S3Error as exc:
            logger.error(
                "MinIO 查找对象失败 bucket={} document={} code={} err={}",
                self._bucket,
                document_id,
                exc.code,
                exc.message,
            )
            raise MinioError(
                f"MinIO 查找对象失败 document={document_id}: "
                f"{str(exc.code)} {str(exc.message)}",
                code=str(exc.code),
            ) from exc

        if info is None:
            logger.info("MinIO 未找到文档 document={}", document_id)
        return info

    @staticmethod
    def _get_meta_value(metadata: dict[str, str], name: str) -> str | None:
        """从对象元数据里取一个自定义值（S3 元数据键大小写不敏感）。

        写入时值经过百分号编码（见 :meth:`_build_metadata`），这里 ``unquote`` 还原；
        纯 ASCII 的值往返不变。

        Args:
            metadata: ``stat_object`` 返回的元数据。
            name: 自定义元数据名（不含 ``X-Amz-Meta-`` 前缀），如 ``id``。

        Returns:
            写入过该键时返回其值，否则返回 ``None``。
        """
        target = f"x-amz-meta-{name}"
        for key, value in metadata.items():
            if key.lower() == target and value:
                return unquote(str(value))
        return None

    async def delete_file(self, object_name: str) -> None:
        """删除对象（幂等：对象不存在时 S3 同样返回成功）。

        Args:
            object_name: 要删除的对象键。

        Raises:
            ValueError: ``object_name`` 为空。
            MinioError: 存储桶不可用，或删除过程中发生 S3 错误。
        """
        if not object_name:
            raise ValueError("delete_file 必须显式提供 object_name")

        await self.ensure_bucket()
        try:
            await asyncio.to_thread(
                self.client.remove_object, self._bucket, object_name
            )
        except S3Error as exc:
            logger.error(
                "MinIO 删除对象失败 bucket={} object={} code={} err={}",
                self._bucket,
                object_name,
                exc.code,
                exc.message,
            )
            raise MinioError(
                f"MinIO 删除对象失败 object={object_name}: "
                f"{str(exc.code)} {str(exc.message)}",
                code=str(exc.code),
            ) from exc

        logger.info(
            "MinIO 删除对象完成 bucket={} object={}", self._bucket, object_name
        )

    @staticmethod
    def guess_content_type(filename: str) -> str:
        """按扩展名推断 MIME 类型，无法推断时回退 ``application/octet-stream``。"""
        content_type, _ = mimetypes.guess_type(filename)
        return content_type or DEFAULT_CONTENT_TYPE
