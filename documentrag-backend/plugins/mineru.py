"""MinerU v4 API 异步客户端插件（单文件解析，内容进出、不落盘）。

官方接口文档：https://mineru.net/apiManage/docs

MinerU v4 本地文件无独立的「单文件」端点——文件上传必经 ``/file-urls/batch``
批量入口（``files`` 数组），但本插件**对外仅暴露单文件解析**：批量逻辑隐藏在内部，
方便上游按业务节奏逐文件调度。

工作流程：

    ┌──────────────────────────────────────────────────────────────────┐
    │ 1. POST /file-urls/batch    （files 数组仅含 1 项）               │
    │    请求体：JSON（files 元数据 + model_version / language / ...）   │
    │    响应  ：batch_id + file_urls[]（单条 OSS 预签名上传链接）       │
    │                                                                  │
    │ 2. PUT <file_urls[0]>                                            │
    │    请求体：调用方给的**文件字节**（不设 Content-Type）             │
    │    副作用：MinerU 自动提交解析任务                                 │
    │                                                                  │
    │ 3. GET /extract-results/batch/{batch_id}（轮询）                  │
    │    终态：done（含 full_zip_url）/ failed（含 err_msg）             │
    │                                                                  │
    │ 4. GET <full_zip_url>                                            │
    │    压缩包只在内存中解压，不落盘                                     │
    │    抽取 full.md   → ParsedDocument.markdown（字节）                │
    │    抽取 images/*  → ParsedDocument.images（文件名 → 字节）         │
    └──────────────────────────────────────────────────────────────────┘

**本插件不写磁盘**：解析结果以 :class:`ParsedDocument` 返回，落盘由调用方决定
（入库链路写到 ``storage/<id>/``）。这样调用方可以先读内容、等解析成功后再把
原件与 Markdown 一起归档，失败的解析不会在存储里留下半成品。

使用::

    client = Mineru()                        # 进程内单例
    await client.register()
    content = Path("docs/spec.pdf").read_bytes()
    parsed = await client.parse_content(content, "spec.pdf", data_id="6f1c...")
    Path("storage/6f1c.../spec.md").write_bytes(parsed.markdown)
    await client.close()

日志：通过 :mod:`logger` 暴露的 **loguru** 实例输出（控制台 + 按天轮转文件），
统一风格——

- ``debug``：轮询节拍、HTTP 入参关键字段、zip 目录清单
- ``info`` ：阶段推进（上传链接、上传完成、解析完成、抽取结果）
- ``warning``：Token 缺失
- ``error``：业务码非零、轮询超时、解析失败、zip 缺关键文件

配置：
    ``[tool.mineru]`` 中的 ``token`` 与 ``base_url`` 会被 :mod:`config`
    读取为 ``MINERU_TOKEN`` / ``MINERU_BASE_URL``。
"""

from __future__ import annotations

import asyncio
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

import config
from logger import logger
from plugins import Plugin

__all__ = ["IMAGES_DIRNAME", "Mineru", "MineruAPIError", "ParsedDocument"]

# MinerU v4 终态集合：到达即停止轮询
_TERMINAL_STATES = frozenset({"done", "failed"})
# zip 内 Markdown 文件名（官方固定输出 ``full.md``）
_MARKDOWN_ENTRY = "full.md"
# zip 内图片目录前缀（MinerU 输出约定：``images/<文件名>``）
_IMAGES_ENTRY_PREFIX = "images/"

#: 图片子目录名：调用方按它落盘，Markdown 里的 ``images/...`` 相对引用才成立
IMAGES_DIRNAME = "images"


class MineruAPIError(RuntimeError):
    """MinerU 业务错误：HTTP 200 但 ``code != 0``，或 zip 缺关键文件。"""

    def __init__(self, message: str, *, code: int | None = None, msg: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.msg = msg


@dataclass(frozen=True)
class ParsedDocument:
    """MinerU 的解析结果（尚未落盘）：Markdown 正文与随包图片。"""

    #: ``full.md`` 的内容
    markdown: bytes
    #: 结果包内 ``images/`` 下的图片：文件名 → 图片字节（与 Markdown 中的相对引用同名）
    images: dict[str, bytes] = field(default_factory=dict)


class Mineru(Plugin):
    """MinerU v4 解析 API 异步客户端（仅处理单条文档，结果不落盘）。

    调用 :meth:`parse_content` 即可完成「上传 → 轮询 → 下载 zip → 抽取内容」全流程，
    返回 :class:`ParsedDocument`；落盘与归档由调用方负责。
    """

    DEFAULT_MODEL_VERSION = "vlm"
    DEFAULT_POLL_INTERVAL = 5.0
    DEFAULT_POLL_TIMEOUT = 900.0
    DEFAULT_LANGUAGE = "ch"

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    async def register(self) -> None:
        """创建异步 HTTP 客户端；Token 缺失仅告警，不阻断启动（fail-open）。

        与 ``plugins/celery/__init__.py`` 中 ``CeleryWorker`` 的策略一致：启动失败不阻断 API，便于
        上线时即暴露配置问题而非隐藏。
        """
        if not config.MINERU_TOKEN:
            logger.warning(
                "MINERU_TOKEN 未配置，MinerU 调用将返回 401/403。"
                "请在 pyproject.toml 的 [tool.mineru] 配置 token 后再调用。"
            )
        self._client = httpx.AsyncClient(
            base_url=config.MINERU_BASE_URL,
            headers={
                "Authorization": f"Bearer {config.MINERU_TOKEN}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=60.0, pool=10.0),
        )
        logger.info("MinerU 客户端已就绪 base_url={}", config.MINERU_BASE_URL)

    async def close(self) -> None:
        """关闭 HTTP 客户端（幂等：重复关闭安全）。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("MinerU 客户端已关闭")

    @property
    def client(self) -> httpx.AsyncClient:
        """惰性取客户端；未注册时抛错，避免在请求处理里埋雷。"""
        if self._client is None:
            raise RuntimeError(
                "MinerU 客户端未注册：请确认 plugins 已随 FastAPI lifespan 启动，"
                "或先显式调用 Mineru().register()"
            )
        return self._client

    # ------------------------------------------------------------------ #
    # 对外主入口
    # ------------------------------------------------------------------ #
    async def parse_file(self, file_path: str | Path) -> ParsedDocument:
        """读取本地文件并交给 :meth:`parse_content` 解析（结果同样不落盘）。

        Args:
            file_path: 本地文档/图片路径（≤200 MB、≤200 页）。

        Returns:
            解析结果（Markdown 字节 + 图片字节），由调用方决定落盘位置。

        Raises:
            FileNotFoundError: ``file_path`` 不存在。
            MineruAPIError: API 业务错误（code != 0 / 解析失败 / 轮询超时
                / 批量结果为空 / zip 缺关键文件）。
            httpx.HTTPError: HTTP 传输错误（含网络中断、超时）。
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")
        content = await asyncio.to_thread(path.read_bytes)
        return await self.parse_content(content, path.name)

    async def parse_content(
            self,
            content: bytes,
            filename: str,
            *,
            data_id: str | None = None,
    ) -> ParsedDocument:
        """提交文档内容到 MinerU 解析，返回 Markdown 与图片内容（不落盘）。

        Args:
            content: 文档字节（如 PDF 原文）；调用方先读取内容再提交，解析成功后由
                调用方归档，失败的解析不会在存储里留下半成品。
            filename: 文件名，用于 MinerU 侧的展示与类型识别（只取纯文件名部分）。
            data_id: 提交给 MinerU 的任务标识；缺省取文件名的前缀（不含扩展名）。
                入库链路传文档 ID，便于在 MinerU 控制台对照。

        Returns:
            :class:`ParsedDocument`：Markdown 正文与随包图片。

        Raises:
            MineruAPIError: API 业务错误（code != 0 / 解析失败 / 轮询超时
                / 批量结果为空 / zip 缺关键文件）。
            httpx.HTTPError: HTTP 传输错误（含网络中断、超时）。
        """
        name = Path(filename).name
        did = data_id or Path(name).stem
        logger.info(
            "MinerU 准备解析文档 file={} data_id={} size={}B", name, did, len(content)
        )

        # 1. 申请上传链接（files 数组仅含 1 项——v4 本地文件唯一入口）
        body: dict[str, Any] = {
            "files": [{"name": name, "data_id": did}],
            "model_version": self.DEFAULT_MODEL_VERSION,
            "is_ocr": False,
            "enable_formula": True,
            "enable_table": True,
            "language": self.DEFAULT_LANGUAGE,
        }
        data = await self._post_json("/file-urls/batch", body)
        batch_id: str = data["batch_id"]
        upload_urls: list[str] = data["file_urls"]
        logger.info("MinerU 已申请上传链接 batch_id={} file={}", batch_id, name)

        if len(upload_urls) != 1:
            logger.error(
                "MinerU 上传链接数量异常 batch_id={} expected=1 actual={}",
                batch_id, len(upload_urls),
            )
            raise MineruAPIError(
                f"单文件上传期望返回 1 条预签名链接，实际 {len(upload_urls)} 条"
            )

        # 2. PUT 文件内容到预签名 URL
        await self._put_content(upload_urls[0], content, name)
        logger.info("MinerU 文件上传完成 batch_id={}，等待解析", batch_id)

        # 3. 轮询批量结果（数组仅含本文件一项）
        results = await self._poll_batch(
            batch_id,
            self.DEFAULT_POLL_INTERVAL,
            self.DEFAULT_POLL_TIMEOUT,
        )

        if not results:
            logger.error("MinerU 批量结果为空 batch_id={}", batch_id)
            raise MineruAPIError(
                f"MinerU 批量结果为空 batch_id={batch_id}",
                msg="empty extract_result",
            )

        # 4. 校验状态
        item = results[0]
        state: str = item.get("state", "")
        err_msg: str = item.get("err_msg", "")
        zip_url: str = item.get("full_zip_url", "")
        file_name: str = item.get("file_name", "") or name

        if state != "done" or not zip_url:
            logger.error(
                "MinerU 解析失败 file={} state={} err={}",
                file_name, state, err_msg,
            )
            raise MineruAPIError(
                f"MinerU 解析失败 file={file_name} state={state} err={err_msg}",
                msg=err_msg,
            )

        # 5. 下载结果包并在内存中抽取 Markdown 与图片（不落盘）
        parsed = await self._download_result(zip_url)
        logger.info(
            "MinerU 解析完成 file={} markdown={}B images={}",
            file_name, len(parsed.markdown), len(parsed.images),
        )
        return parsed

    # ------------------------------------------------------------------ #
    # 内部：HTTP / 轮询 / 抽取
    # ------------------------------------------------------------------ #
    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON，自动检查 ``code == 0``；非零抛 :class:`MineruAPIError`."""
        logger.debug(
            "MinerU POST {} payload_keys={}", path, sorted(payload.keys())
        )
        resp = await self.client.post(path, json=payload)
        resp.raise_for_status()
        body = resp.json()
        code = body.get("code", 0)
        if code != 0:
            logger.error(
                "MinerU POST {} 业务错误 code={} msg={}",
                path, code, body.get("msg"),
            )
            raise MineruAPIError(
                f"MinerU {path} 返回错误 code={code} msg={body.get('msg')}",
                code=code,
                msg=body.get("msg", ""),
            )
        return body["data"]

    async def _get_json(self, path_or_url: str) -> dict[str, Any]:
        """GET JSON；非零业务码抛 :class:`MineruAPIError`."""
        resp = await self.client.get(path_or_url)
        resp.raise_for_status()
        body = resp.json()
        code = body.get("code", 0)
        if code != 0:
            logger.error(
                "MinerU GET {} 业务错误 code={} msg={}",
                path_or_url, code, body.get("msg"),
            )
            raise MineruAPIError(
                f"MinerU GET {path_or_url} 返回错误 code={code} msg={body.get('msg')}",
                code=code,
                msg=body.get("msg", ""),
            )
        return body["data"]

    async def _put_content(self, url: str, content: bytes, filename: str) -> None:
        """PUT 文件内容到预签名 URL：``Content-Type`` 必须为空。

        ``httpx.Headers`` 中空字符串代表「删除该请求头」；预签名 URL 上传
        校验不要求 ``Content-Type``，参照官方示例 ``requests.put(url, data=f)``。
        """
        logger.info("MinerU 开始上传 file={} size={}B", filename, len(content))
        resp = await self.client.put(
            url,
            content=content,
            headers={"Content-Type": ""},
        )
        resp.raise_for_status()
        logger.info(
            "MinerU 上传完成 file={} status={}",
            filename, resp.status_code,
        )

    async def _poll_batch(
            self, batch_id: str, interval: float, timeout: float
    ) -> list[dict[str, Any]]:
        """轮询批量任务直到唯一一项 ``done`` / ``failed``。"""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        attempt = 0
        while True:
            data = await self._get_json(f"/extract-results/batch/{batch_id}")
            results: list[dict[str, Any]] = data.get("extract_result", [])
            states = [r.get("state", "") for r in results]
            attempt += 1
            if results and all(s in _TERMINAL_STATES for s in states):
                logger.info(
                    "MinerU 任务完成 batch_id={} states={} attempts={}",
                    batch_id, states, attempt,
                )
                return results
            if loop.time() > deadline:
                logger.error(
                    "MinerU 轮询超时 batch_id={} timeout={}s attempts={}",
                    batch_id, timeout, attempt,
                )
                raise TimeoutError(
                    f"MinerU 批量轮询超时 batch_id={batch_id} timeout={timeout}s"
                )
            logger.debug(
                "MinerU 轮询 batch_id={} attempt={} states={}",
                batch_id, attempt, states,
            )
            await asyncio.sleep(interval)

    async def _download_result(self, zip_url: str) -> ParsedDocument:
        """下载结果压缩包，在内存中解压并抽取 ``full.md`` 与图片。

        压缩包**不落盘**：响应体直接交给 ``zipfile`` 在 ``io.BytesIO`` 上解压，
        不产生任何临时文件，用完随响应对象一起释放。

        Args:
            zip_url: MinerU 返回的结果包下载链接（CDN 绝对 URL）。

        Returns:
            解析结果（Markdown 字节 + 图片字节）。

        Raises:
            MineruAPIError: 压缩包内缺少 ``full.md``。
            httpx.HTTPError: 下载失败。
        """
        logger.info("MinerU 下载结果压缩包 url={}", zip_url)
        # zip 下载链接是 CDN 域名（绝对 URL），不会被 ``base_url`` 干扰
        resp = await self.client.get(zip_url)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            logger.debug("MinerU zip entries={}", names)
            if _MARKDOWN_ENTRY not in names:
                logger.error(
                    "MinerU zip 缺关键文件 expected={} actual={}",
                    _MARKDOWN_ENTRY, names,
                )
                raise MineruAPIError(
                    f"zip 包内未找到 {_MARKDOWN_ENTRY}：entries={names}"
                )
            markdown_bytes = zf.read(_MARKDOWN_ENTRY)
            images = self._extract_images(zf)

        logger.info(
            "MinerU 结果已抽取 markdown={}B images={}",
            len(markdown_bytes), len(images),
        )
        return ParsedDocument(markdown=markdown_bytes, images=images)

    @staticmethod
    def _extract_images(zf: zipfile.ZipFile) -> dict[str, bytes]:
        """把 zip 内 ``images/`` 下的图片读成 ``{文件名: 字节}``。

        MinerU 结果包以 ``images/<文件名>`` 组织图片，``full.md`` 中按同名相对
        路径引用；这里按文件名平铺（图片名带内容哈希，正常不会重名）。

        Args:
            zf: 已打开的结果压缩包。

        Returns:
            图片文件名 → 字节；结果包内无图片时为空字典。
        """
        images = {}
        for entry in zf.namelist():
            # 跳过目录条目（``images/``）与非图片路径
            if not entry.startswith(_IMAGES_ENTRY_PREFIX) or entry.endswith("/"):
                continue
            filename = Path(entry).name
            if not filename:
                continue
            images[filename] = zf.read(entry)
        logger.info("MinerU 已抽取图片 {} 张", len(images))
        return images
