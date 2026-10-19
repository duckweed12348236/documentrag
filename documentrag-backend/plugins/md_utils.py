"""Markdown 文档的解析与切分工具（基于 langchain-text-splitters 与 markdown-it-py）。

文档切分面向 RAG 入库场景，提供两个工具函数：

1. :func:`split_by_header`——按 Markdown 标题把整篇文档切成章节，返回
   ``[(标题, 标题行号, 内容), ...]``；内容不含标题行，标题与行号由元组前两项
   单独承载，无标题的前言部分标题为空串、行号为 0。
2. :func:`split`——在章节切分的基础上，按 **token**（tiktoken ``cl100k_base``
   编码）计数把内容整理成多个文本块，供向量化入库；每个块带上所属章节的标题行，
   便于检索时保留上下文。

长度约束（:func:`split` 的参数，单位均为 token）：

- ``min_length``：块的期望最小 token 数。相邻短块会尽量合并，避免碎片块；
- ``soft_max_length``：块的期望最大 token 数。二次切分与合并都以它为界；
- ``hard_max_length``：块的绝对 token 上限，极端情况下的兜底切断。

切分规则：

- 章节内容超过软上限时，用 :class:`MarkdownTextSplitter`
  （``from_tiktoken_encoder``，即以 token 计数）二次切分，每个片段都重新带上
  所属标题行；
- 相邻块合并后不超过软上限时合并；
- 仍超过硬上限的块（极端情况）按 token 上限强制切断，断点尽量落在换行处。

说明：标题切分用 :class:`MarkdownHeaderTextSplitter`（内置代码围栏与标题嵌套
处理），但其产物不含行号，故另行扫描原文标题行，按顺序与章节一一对应得到
行号（1 起计）。

使用::

    sections = split_by_header(markdown)   # [(标题, 行号, 内容), ...]
    chunks = split(markdown)               # ["## 标题\n\n正文", ...]
    chunks = split(markdown, min_length=256, soft_max_length=512)  # 可调 token 约束

Markdown 结构工具
=================

除切分器外，本模块还基于 **markdown-it-py** 提供一组文档结构工具：

- :func:`find_images`：查找 Markdown 图片（``![alt](src "title")``），可用
  ``context`` 参数附带图片标记前后各 n 个字符的上下文；
- :func:`find_blocks`：查找表格（Markdown 管道表格与 HTML ``<table>``），同样
  可用 ``context`` 参数附带表格前后各 n 个字符的上下文；
- :func:`replace`：按原文偏移批量替换若干 ``[start, end)`` 区间。

命中项都带 ``start`` / ``end`` 字符偏移（左闭右开）与 ``raw`` 原文片段，可直接
用于定位与替换。已解析出的代码区域（围栏、缩进代码块）中的伪语法不会被命中。

:func:`find_blocks` 只返回**互不重叠的表格**（嵌套表格只保留最外层），这样结果
可直接喂给 :func:`replace` 批量替换而不会产生区间重叠。

批量替换的偏移一律基于**同一份原文**（即 :func:`find_images` /
:func:`find_blocks` 返回的偏移），:func:`replace` 内部按位置从后往前落笔，因此
前面区间的偏移不会因后面文本长度变化而失效。

已知限制：引用式图片（``![alt][ref]``）、以及写在行内代码里的伪语法无法定位，
这类命中会被跳过。

使用::

    for image in find_images(markdown, context=100):   # 带上下文的命中项
        print(image.src, image.before, image.after)
    markdown = replace(markdown, [                      # 批量替换全部图片
        (image.start, image.end, f"![{describe(image)}]({image.src})")
        for image in find_images(markdown, context=100)
    ])
    markdown = replace(markdown, [                      # 批量替换全部表格
        (table.start, table.end, summarize(table.raw))
        for table in find_blocks(markdown)
    ])

    for table in find_blocks(markdown, context=100):    # 带上下文的表格
        print(table.before, table.headers, table.after)
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace as dataclass_replace
from functools import lru_cache
from urllib.parse import unquote

import tiktoken
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter
from markdown_it import MarkdownIt

__all__ = [
    "MdImage",
    "MdTable",
    "find_tables",
    "find_images",
    "replace",
    "split",
    "split_by_header",
]


@dataclass(frozen=True)
class _Section:
    """章节内部结构：标题、标题行号、标题层级、正文。"""

    title: str
    line_no: int
    level: int
    body: str


# ---------------------------------------------------------------------- #
# 文档切分（基于 langchain-text-splitters）
# ---------------------------------------------------------------------- #
#: tiktoken 编码名；用于 token 计数与切分
_ENCODING_NAME = "cl100k_base"
#: 标题切分规则：``(Markdown 前缀, 元数据键)``，顺序即层级由浅到深
_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
    ("#####", "h5"),
    ("######", "h6"),
]

_TOKENIZER = tiktoken.get_encoding(_ENCODING_NAME)
#: 块间连接符 ``\n\n`` 占用的 token 数，用于合并时的长度预估
_SEPARATOR_TOKENS = len(_TOKENIZER.encode("\n\n"))
# 按标题切分：strip_headers=True 使正文不含标题行，标题信息落在 metadata
_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=_HEADERS_TO_SPLIT_ON,
    strip_headers=True,
)


def split_by_header(text: str) -> list[tuple[str, int, str]]:
    """按 Markdown 标题把文档切成章节。

    Args:
        text: Markdown 原文。

    Returns:
        ``[(标题, 标题行号, 内容), ...]``：

        - ``标题``：该章节最深一级标题的文本（如 ``三级标题``）；文档开头
          无标题的前言部分为空串；
        - ``标题行号``：该标题在原文中的行号（1 起计）；前言部分为 0；
        - ``内容``：不含标题行的章节正文（已去除首尾空白），空正文的标题
          不会单独成项。
    """
    return [
        (section.title, section.line_no, section.body)
        for section in _sections(text)
    ]


def split(
    text: str,
    min_length: int = 512,
    soft_max_length: int = 1024,
    hard_max_length: int = 2048,
) -> list[str]:
    """把文档先按标题切成章节，再按 token 把章节整理成多个文本块。

    每个块带上所属章节的标题行（前言块除外），便于后续检索保留上下文。

    Args:
        text: Markdown 原文。
        min_length: 块的期望最小 token 数，短于该值的相邻块会被尽量合并。
        soft_max_length: 块的期望最大 token 数。
        hard_max_length: 块的绝对 token 上限。

    Returns:
        文本块列表；空文档返回空列表。

    Raises:
        ValueError: 长度参数不满足
            ``0 < min_length <= soft_max_length <= hard_max_length``。
    """
    if not 0 < min_length <= soft_max_length <= hard_max_length:
        raise ValueError(
            "长度约束需满足 0 < min_length <= soft_max_length <= hard_max_length，"
            f"实际为 min_length={min_length} "
            f"soft_max_length={soft_max_length} "
            f"hard_max_length={hard_max_length}"
        )

    atoms = []
    for section in _sections(text):
        # 标题行（无标题的前言为空串），block 内补上以保留章节上下文
        heading = ""
        if section.title and section.level > 0:
            heading = f"{'#' * min(section.level, len(_HEADERS_TO_SPLIT_ON))} {section.title}"
        # 标题行 + 连接符的 token 占用，从软上限中扣掉，避免加上标题后超限
        reserved = _count_tokens(heading) + _SEPARATOR_TOKENS if heading else 0
        budget = max(soft_max_length - reserved, 1)

        pieces = [section.body]
        if _count_tokens(section.body) > budget:
            pieces = _token_splitter(budget).split_text(section.body)

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            unit = f"{heading}\n\n{piece}" if heading else piece
            atoms.append((unit, _count_tokens(unit)))

    return _enforce_hard_limit(
        _merge(atoms, min_length, soft_max_length), hard_max_length
    )

# ---------------------------------------------------------------------- #
# 内部：章节解析
# ---------------------------------------------------------------------- #
def _sections(text: str) -> list[_Section]:
    """用 :class:`MarkdownHeaderTextSplitter` 切章节，并补上标题行号。"""
    if not text or not text.strip():
        return []

    headings = _scan_headings(text)
    sections = []
    cursor = 0
    for document in _HEADER_SPLITTER.split_text(text):
        body = document.page_content.strip()
        if not body:
            continue

        # metadata 是当前标题栈，取最深一级标题作为该章节的标题
        title, level = "", 0
        for depth in range(len(_HEADERS_TO_SPLIT_ON), 0, -1):
            candidate = document.metadata.get(_HEADERS_TO_SPLIT_ON[depth - 1][1])
            if isinstance(candidate, str) and candidate.strip():
                title, level = candidate.strip(), depth
                break

        line_no = 0
        if title:
            # 章节与标题行都按原文顺序出现，顺序匹配即可定位（跳过被前序章节
            # 占用或未产出内容的标题）
            for index in range(cursor, len(headings)):
                head_line, head_level, head_title = headings[index]
                if head_level == level and head_title == title:
                    line_no = head_line
                    cursor = index + 1
                    break

        sections.append(_Section(title=title, line_no=line_no, level=level, body=body))
    return sections


def _scan_headings(text: str) -> list[tuple[int, int, str]]:
    """扫描原文定位全部标题行。

    判定规则与 :class:`MarkdownHeaderTextSplitter` 一致：行首为 ``#``~``######``
    且其后为空或空格，且不处于代码围栏内。

    Args:
        text: Markdown 原文。

    Returns:
        ``[(行号, 层级, 标题), ...]``，行号 1 起计，顺序与原文一致。
    """
    # 前缀按长度倒序，避免 ``###`` 被 ``#`` 抢先匹配
    prefixes = sorted(_HEADERS_TO_SPLIT_ON, key=lambda item: len(item[0]), reverse=True)
    headings = []
    in_code_block = False
    fence = ""
    for line_no, raw_line in enumerate(text.split("\n"), start=1):
        line = raw_line.strip()
        if not in_code_block:
            if line.startswith("```") and line.count("```") == 1:
                in_code_block, fence = True, "```"
                continue
            if line.startswith("~~~"):
                in_code_block, fence = True, "~~~"
                continue
        elif line.startswith(fence):
            in_code_block, fence = False, ""
            continue

        if in_code_block or not line:
            continue

        for prefix, _key in prefixes:
            if not line.startswith(prefix):
                continue
            if len(line) == len(prefix) or line[len(prefix)] == " ":
                headings.append((line_no, len(prefix), line[len(prefix):].strip()))
                break
    return headings

# ---------------------------------------------------------------------- #
# 内部：token 计数 / 切分 / 合并
# ---------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _token_splitter(chunk_size: int) -> MarkdownTextSplitter:
    """取指定 token 上限的切分器（按上限缓存）。

    :class:`MarkdownTextSplitter` 复用 Markdown 分隔符优先级，配合
    ``from_tiktoken_encoder`` 以 token 计数决定切点，断点落在分隔符 / 字符边界，
    不会切断多字节字符。
    """
    return MarkdownTextSplitter.from_tiktoken_encoder(
        encoding_name=_ENCODING_NAME,
        chunk_size=chunk_size,
        chunk_overlap=0,
    )


def _count_tokens(text: str) -> int:
    """统计文本的 token 数（tiktoken ``cl100k_base``）。"""
    if not text:
        return 0
    return len(_TOKENIZER.encode(text))


def _merge(atoms: list[tuple[str, int]], min_length: int, soft_max_length: int) -> list[str]:
    """贪心合并相邻短块：块 token 数不足 ``min_length`` 且合并后不超软上限时合并。"""
    if not atoms:
        return []

    blocks = []
    buffer, buffer_tokens = atoms[0]
    for atom, atom_tokens in atoms[1:]:
        if (
                buffer_tokens < min_length
                and buffer_tokens + atom_tokens + _SEPARATOR_TOKENS <= soft_max_length
        ):
            buffer = f"{buffer}\n\n{atom}"
            buffer_tokens += atom_tokens + _SEPARATOR_TOKENS
            continue
        blocks.append(buffer)
        buffer, buffer_tokens = atom, atom_tokens
    blocks.append(buffer)
    return blocks


def _enforce_hard_limit(blocks: list[str], hard_max_length: int) -> list[str]:
    """兜底：把超过硬上限的块按 token 上限强制切断。"""
    result = []
    for block in blocks:
        if _count_tokens(block) <= hard_max_length:
            result.append(block)
            continue
        result.extend(_hard_cut(block, hard_max_length))
    return result


def _hard_cut(block: str, hard_max_length: int) -> list[str]:
    """按 token 上限逐段切断单个超长块，断点尽量落在换行处。

    用二分查找定位「token 数恰好不超过硬上限」的最长前缀，确保不会切断
    多字节字符。
    """
    chunks = []
    remaining = block
    while _count_tokens(remaining) > hard_max_length:
        low, high = 1, len(remaining)
        while low < high:
            mid = (low + high + 1) // 2
            if _count_tokens(remaining[:mid]) <= hard_max_length:
                low = mid
            else:
                high = mid - 1
        cut = remaining.rfind("\n", 0, low)
        if cut <= 0:
            cut = low
        piece = remaining[:cut].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


# ====================================================================== #
# Markdown 结构工具（基于 markdown-it-py）
# ====================================================================== #
#: Markdown 图片语法 ``![alt](src "title")``：src 支持 ``<>`` 包裹与 ``\`` 转义
_IMAGE_PATTERN = re.compile(
    r"!\[(?P<alt>.*?)]\(\s*"
    r"(?P<src><[^<>\n]*>|(?:[^\s()\\]|\\.)+)"
    r"(?:\s+(?P<title>\"[^\"]*\"|'[^']*'|\([^()]*\)))?\s*\)"
)
#: HTML 表格标签（非贪婪，不处理嵌套表格）
_TABLE_TAG_PATTERN = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)
#: 反斜杠转义
_ESCAPE_PATTERN = re.compile(r"\\(.)")
#: markdown-it 解析器（CommonMark + 表格 + 删除线）
_PARSER = MarkdownIt("commonmark").enable(["table", "strikethrough"])


@dataclass(frozen=True)
class MdImage:
    """文档中命中的 Markdown 图片。

    Attributes:
        src: 图片地址。
        alt: 替代文本（``![alt](src)`` 中的 ``alt``）。
        title: 标题；无标题时为空串。
        start: 在原文中的起始字符偏移（含）。
        end: 在原文中的结束字符偏移（不含）。
        raw: 命中的原文片段，可直接用于替换或落库。
        before: 图片标记前若干字符的上下文；未启用上下文时为 ``""``。
        after: 图片标记后若干字符的上下文；未启用上下文时为 ``""``。
    """

    src: str
    alt: str
    title: str
    start: int
    end: int
    raw: str
    before: str = ""
    after: str = ""


@dataclass(frozen=True)
class MdTable:
    """文档中命中的表格。

    Attributes:
        start: 在原文中的起始字符偏移（含）。
        end: 在原文中的结束字符偏移（不含）。
        raw: 命中的原文片段（HTML 表格即整段 ``<table>...</table>``）。
        is_html: 是否来自 HTML ``<table>`` 标签（否则为 Markdown 管道表格）。
        headers: Markdown 表格的表头单元格文本；HTML 表格为空列表。
        rows: Markdown 表格的数据行（每行一个单元格文本列表）；HTML 表格为空。
        before: 表格前若干字符的上下文；未启用上下文时为 ``""``。
        after: 表格后若干字符的上下文；未启用上下文时为 ``""``。
    """

    start: int
    end: int
    raw: str
    is_html: bool
    headers: list[str]
    rows: list[list[str]]
    before: str = ""
    after: str = ""


@dataclass
class _Analysis:
    """一次解析的中间结果，供各查找函数复用。"""

    inline_spans: list[tuple[int, int]]
    html_spans: list[tuple[int, int]]
    images: list[tuple[str, str, str]]
    tables: list[MdTable]


def find_images(text: str, context: int | None = None) -> list[MdImage]:
    """查找 Markdown 文档中的图片（``![alt](src "title")``）。

    结果按原文位置升序排列；可选返回图片标记前后各若干字符的上下文，便于把
    邻近正文一并交给下游（如图像描述模型）。

    Args:
        text: Markdown 原文。
        context: 需要一并返回的上下文长度（字符）。传入正数时，``MdImage.before``
            与 ``MdImage.after`` 分别取图片标记前 / 后各 ``context`` 个字符（原样
            截取，不裁剪空白）；``None``（默认）或非正数表示不返回上下文。

    Returns:
        :class:`MdImage` 列表；无命中时为空列表。

    Note:
        仅处理 Markdown 图片语法。引用式图片（``![alt][ref]``）无法在原文中
        定位，会被跳过；写在行内代码里的伪语法会被忽略。
    """
    if not text:
        return []

    size = max(context or 0, 0)
    analysis = _analyze(text)
    # markdown-it 解析出的 src 已被归一化（含百分号编码），统一解码后与原文比对
    keys = [
        unquote(html.unescape(_ESCAPE_PATTERN.sub(r"\1", src)))
        for src, _alt, _title in analysis.images
    ]

    images = []
    cursor = 0
    # 候选语法与解析出的 image token 均按原文顺序出现，按归一化后的 src 顺序
    # 配对，可越过行内代码等未被解析成图片的伪语法
    for match in _IMAGE_PATTERN.finditer(text):
        start, end = match.start(), match.end()
        if not any(begin <= start < stop for begin, stop in analysis.inline_spans):
            continue
        src = match.group("src").strip()
        if src.startswith("<") and src.endswith(">"):
            src = src[1:-1]
        src = unquote(html.unescape(_ESCAPE_PATTERN.sub(r"\1", src)))
        for index in range(cursor, len(analysis.images)):
            if keys[index] != src:
                continue
            token_src, alt, title = analysis.images[index]
            images.append(MdImage(
                src=token_src,
                alt=alt,
                title=title,
                start=start,
                end=end,
                raw=match.group(0),
                before=text[max(0, start - size):start],
                after=text[end:end + size],
            ))
            cursor = index + 1
            break

    images.sort(key=lambda item: item.start)
    return images


def find_tables(text: str, context: int | None = None) -> list[MdTable]:
    """查找 Markdown 文档中的表格。

    同时识别 Markdown 管道表格（解析出表头与数据行）与 HTML ``<table>`` 区块
    （只提供 ``raw``，不做 HTML 解析），结果按原文位置升序排列；可选返回表格
    前后各若干字符的上下文，便于把邻近正文一并交给下游（如表格摘要模型）。

    Args:
        text: Markdown 原文。
        context: 需要一并返回的上下文长度（字符）。传入正数时，``before`` 与
            ``after`` 分别取表格前 / 后各 ``context`` 个字符（原样截取，不裁剪
            空白）；``None``（默认）或非正数表示不返回上下文。

    Returns:
        按原文位置升序排列的 :class:`MdTable` 列表；无命中时为空列表。

    Note:
        只返回**互不重叠的表格**（嵌套表格只保留最外层），这样结果可直接交给
        :func:`replace` 批量替换而不会产生区间重叠。
    """
    if not text:
        return []

    analysis = _analyze(text)
    blocks: list[MdTable] = list(analysis.tables)
    # HTML 表格在正文 / HTML 区块内按标签定位，只看整段被当作 HTML 解析的区域
    for match in _TABLE_TAG_PATTERN.finditer(text):
        start, end = match.start(), match.end()
        if not any(begin <= start < stop for begin, stop in analysis.html_spans):
            continue
        blocks.append(MdTable(
            start=start,
            end=end,
            raw=match.group(0),
            is_html=True,
            headers=[],
            rows=[],
        ))

    # 起点升序、同起点时范围大的（外层）在前，随后丢掉被外层表格覆盖的表格
    blocks.sort(key=lambda item: (item.start, -item.end))
    blocks_result = []
    for block in blocks:
        if blocks_result and block.start < blocks_result[-1].end:
            continue
        blocks_result.append(block)

    size = max(context or 0, 0)
    if size <= 0:
        return blocks_result
    # 按需补上紧邻的前后文；偏移与 start / end 同源，直接取原文切片
    return [
        dataclass_replace(
            block,
            before=text[max(0, block.start - size):block.start],
            after=text[block.end:block.end + size],
        )
        for block in blocks_result
    ]


def replace(text: str, replacements: Sequence[tuple[int, int, str]]) -> str:
    """按原文偏移批量替换若干 ``[start, end)`` 区间。

    所有 ``start`` / ``end`` 都基于**同一份原文**（如 :func:`find_images` /
    :func:`find_blocks` 给出的偏移）。内部按 ``start`` 从后往前落笔：先替换靠后的
    区间，靠前区间的偏移就不会因后续文本长度变化而失效，因此调用方无需自行排序、
    也无需逐次重算偏移。

    Args:
        text: Markdown 原文。
        replacements: ``(start, end, new_content)`` 序列；``start`` / ``end`` 为
            原文中的字符偏移（左闭右开，允许 ``start == end`` 作纯插入），
            ``new_content`` 为替换后的新内容。

    Returns:
        替换后的文本。

    Raises:
        ValueError: 任一区间不满足 ``0 <= start <= end <= len(text)``，
            或与其它区间重叠（含部分交叉、完全包含）。
    """
    if not text or not replacements:
        return text

    # 从后往前替换：只有比当前区间更靠右的文本会被改写，左侧偏移始终有效
    limit = len(text)
    result = text
    for start, end, new_content in sorted(replacements, key=lambda item: item[0], reverse=True):
        if not 0 <= start <= end <= len(text):
            raise ValueError(f"替换区间越界：[{start}, {end})，原文长度 {len(text)}")
        if end > limit:
            raise ValueError(
                f"替换区间重叠：[{start}, {end}) 与已处理区间的起点 {limit} 冲突"
            )
        result = result[:start] + new_content + result[end:]
        limit = start
    return result


# ---------------------------------------------------------------------- #
# 内部：解析
# ---------------------------------------------------------------------- #
def _analyze(text: str) -> _Analysis:
    """解析文档，收集正文行内区域、HTML 区域、图片 token 与表格。

    行内区域（``inline`` token 的行区间）用于确认 Markdown 图片语法确实位于
    正文中，从而排除代码围栏 / 缩进代码块里的伪语法；HTML 区域（``html_block``
    以及含 ``html_inline`` 的正文区域）用于确认 ``<table>`` 确实被当作 HTML 解析。
    """
    lines = text.split("\n")
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)

    inline_spans = []
    html_spans = []
    images = []
    tables = []
    tokens = _PARSER.parse(text)
    for index, token in enumerate(tokens):
        # 块级 token 的 map 是行区间 [start, end)，换算成源文字符区间 [start, end)
        span = None
        if token.map:
            first, last = token.map[0], token.map[1] - 1
            if 0 <= first <= last < len(lines):
                span = offsets[first], offsets[last] + len(lines[last])

        if token.type == "html_block":
            if span is not None:
                html_spans.append(span)
            continue

        if token.type == "inline":
            if span is not None:
                inline_spans.append(span)
            has_html = False
            for child in token.children or []:
                if child.type == "image":
                    images.append((
                        child.attrGet("src") or "",
                        child.content or "",
                        child.attrGet("title") or "",
                    ))
                elif child.type == "html_inline":
                    has_html = True
            if has_html and span is not None:
                html_spans.append(span)
            continue

        if token.type != "table_open" or span is None:
            continue

        # 读取 table_open..table_close 之间的表头与数据行；单元格文本取 th / td
        # 内 inline token 的 content
        headers: list[str] = []
        rows: list[list[str]] = []
        cells: list[str] | None = None
        in_header = False
        for inner in tokens[index + 1:]:
            if inner.type == "table_close":
                break
            if inner.type == "thead_open":
                in_header = True
            elif inner.type == "thead_close":
                in_header = False
            elif inner.type == "tr_open":
                cells = []
            elif inner.type == "tr_close":
                if in_header:
                    headers.extend(cells or [])  # GFM 表格只有一个表头行
                else:
                    rows.append(cells or [])
                cells = None
            elif inner.type == "inline" and cells is not None:
                cells.append(inner.content.strip())
        tables.append(MdTable(
            start=span[0],
            end=span[1],
            raw=text[span[0]:span[1]],
            is_html=False,
            headers=headers,
            rows=rows,
        ))

    return _Analysis(
        inline_spans=inline_spans,
        html_spans=html_spans,
        images=images,
        tables=tables,
    )
