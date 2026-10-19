"""产品检索接口：检索问答 / 会话列表 / 会话删除。

``POST /retrieval`` 把本轮问题交给检索智能体
（:class:`~plugins.agents.retrieval.graph.Retriever`），把流水线上报的
:class:`~plugins.agents.retrieval.schemas.Step` **以 SSE 逐条推给客户端**——节点一上报就推送，
不再攒到整轮跑完才返回：

- 每条步骤是一个 ``data: <步骤 JSON>`` 事件，全部结束后补一条 ``data: [DONE]``；
  流水线中途异常时先推 ``event: error`` 再结束，前端不必靠「连接断了」猜失败；
- 节点开始上报 ``running``、结束上报 ``success`` / ``error``（失败带 ``error`` 原因）；
- 重排节点的完成步骤带 ``chunks``（本地与网络候选重排、融合后的片段，含 ``source`` /
  ``url`` / ``filename``）；
- 答案生成节点的完成步骤带 ``answer``（正文句末的 ``[序号]`` 即 ``chunks`` 下标 + 1，
  可按同序映射回来源）；
- 最新问题的询问对象不明确、且对话历史中也无法确定时，意图识别节点直接结束流水线，并在
  完成步骤的 ``answer`` 里给出反问用户的语句（前端按普通 ``answer`` 展示即可）。

检索的同时**同步保存会话进度**：每轮问答先开一条轮次
（:meth:`~models.session.Session.open_turn`），流水线每上报一步就用
:meth:`~models.session.Turn.write_stream` 写进数据库（服务端 JSONB 追加），因此回答返回时
历史记录已经完整落库；单步写库失败只记日志、不打断检索（用户不该因为历史写失败拿不到答案），
但开轮次本身的失败会直接报错——会话历史是这次功能的一部分。

会话：请求用 ``session_id``（会话主键）指明延续哪个会话，只给本轮 ``question`` 即可——
上下文（``thread_id`` 对应的短期记忆）由服务端按会话取出，多轮追问中的代词由改写节点补全。
不给 ``session_id`` 即新建会话，新会话的主键通过响应头 ``X-Session-Id`` 返回，后续轮次原样
回传即可；``session_id`` 不存在时返回 404。

- ``GET /retrieval/session`` 列出会话（最新在前）：``id`` / ``thread_id`` / ``title`` /
  ``created_at`` / ``turns``（轮次数，用 ``Count`` 聚合一次查出，不在循环里逐条 count）；
- ``GET /retrieval/session/{session_id}`` 查看会话详情：会话信息 + 每一轮的完整步骤；
- ``DELETE /retrieval/session/{session_id}`` 删除会话（按会话主键）：先清短期记忆
  （checkpointer 的 PostgreSQL checkpoint + Redis 缓存，见 :meth:`Retriever.forget`），
  再删会话记录（轮次随外键级联删除）。

``Retriever`` 是进程内单例插件，由 ``register`` 在应用启动时初始化短期记忆并编译带记忆的
图，因此这里直接实例化即可取到那个已注册的实例。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from tortoise.exceptions import DoesNotExist
from tortoise.functions import Count

from logger import logger
from models.session import Session, Turn
from plugins.agents.retrieval.graph import Retriever
from plugins.agents.retrieval.schemas import Step
from schemas.retrieval import Question

router = APIRouter(prefix="/retrieval")
retriever = Retriever()


@router.post("")
async def retrieve(question: Question) -> StreamingResponse:
    """执行一轮检索问答，把流水线上报的步骤以 SSE 逐条推送。

    会话由请求体里的 ``session_id`` 指明：给了就延续该会话（不存在返回 404），不给就新建
    会话，新会话的主键放在响应头 ``X-Session-Id`` 里返回，后续轮次带上它即可延续上下文。

    本轮问答先落一条轮次记录，随后每一步进度**边落库、边推送**（``Turn.write_stream``）：
    客户端收到第一条事件时，后面的节点可能还在跑，界面可以据此实时显示进度。

    Args:
        question: :class:`~schemas.retrieval.Question`——本轮问题与会话主键；问题缺失或
            去空白后为空时由 FastAPI 直接返回 422。

    Returns:
        ``text/event-stream`` 流（响应头 ``X-Session-Id`` 为本轮所属会话的主键）：

        - ``data: <步骤 JSON>``：每条对应流水线上报的一步（``message`` / ``status`` /
          ``error`` / ``answer`` / ``chunks``）；
        - ``event: error`` + ``data: {"error": ...}``：流水线中途异常（此时不会再发 ``[DONE]``）；
        - ``data: [DONE]``：全部步骤推送完毕。

    Raises:
        HTTPException: 404——``session_id`` 指定的会话不存在；500——开轮次失败（数据库不可用）。
            两者都发生在流开始之前，可正常返回状态码。
    """
    try:
        session, turn = await Session.open_turn(question.content, question.session_id)
    except DoesNotExist as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话不存在：{question.session_id}",
        ) from exc

    # checkpointer 的会话标识按字符串存取：会话上的雪花 ID 转成字符串再交给流水线
    thread_id = str(session.thread_id)
    return StreamingResponse(
        _stream_steps(
            turn.write_stream(retriever.invoke(question.content, thread_id)),
            thread_id,
        ),
        media_type="text/event-stream",
        headers={
            # 客户端据此拿到本轮所属会话的主键（新建会话时尤其需要）
            "X-Session-Id": str(session.id),
            # SSE 禁用缓存，并让 nginx 之类的反向代理别攒满缓冲再吐
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_steps(steps: AsyncIterator[Step], thread_id: str) -> AsyncIterator[str]:
    """把步骤流转成 SSE 文本：一条步骤一个事件，末尾补 ``[DONE]``。

    Args:
        steps: 步骤流（``Turn.write_stream`` 的产物：已落库、原样透传）。
        thread_id: 会话标识，仅用于异常日志。

    Yields:
        SSE 文本块。客户端断开时生成器被取消（``CancelledError`` 直接向外抛），不会把
        断连当成业务失败。
    """
    try:
        async for step in steps:
            yield f"data: {json.dumps(asdict(step), ensure_ascii=False)}\n\n"
    except Exception as exc:  # noqa: BLE001 - 流已开始，只能改用错误事件告知
        logger.opt(exception=exc).error("检索流中断 thread={}", thread_id)
        payload = json.dumps({"error": f"检索失败：{exc}"}, ensure_ascii=False)
        yield f"event: error\ndata: {payload}\n\n"
        return
    yield "data: [DONE]\n\n"


@router.get("/session")
async def list_sessions(
        limit: int | None = Query(None, ge=1, description="最多返回多少条；缺省返回全部"),
) -> list[dict]:
    """列出会话（最新在前），并带上各自的轮次数。

    Args:
        limit: 最多返回多少条；缺省返回全部。

    Returns:
        会话字典列表：``id``（会话主键，详情 / 删除都用它）/ ``thread_id``（短期记忆线程）/
        ``title``（标题，取首轮问题）/ ``created_at``（创建时间，UTC）/ ``turns``（轮次数）。
    """
    # 轮次数用 Count 聚合进同一条查询：会话多时也不会退化成「每条会话一次 count」
    query = Session.annotate(turn_count=Count("turns")).order_by("-id")
    if limit is not None:
        query = query.limit(limit)
    sessions = await query
    return [
        {
            "id": session.id,
            "thread_id": session.thread_id,
            "title": session.title,
            "created_at": session.created_at,
            "turns": session.turn_count,
        }
        for session in sessions
    ]


@router.get("/session/{session_id}")
async def get_session(session_id: int) -> dict:
    """查看会话详情：会话信息 + 每一轮的完整步骤。

    步骤按上报顺序保存在 :attr:`~models.session.Turn.steps` 里（节点开始为 ``running``、
    结束为 ``success`` / ``error``，答案与检索片段分别在对应步骤的 ``answer`` / ``chunks``
    中），因此详情接口直接把落库内容原样返回。

    Args:
        session_id: 会话主键（列表接口返回的 ``id``）。

    Returns:
        ``id`` / ``thread_id`` / ``title`` / ``created_at`` / ``turns``；``turns`` 按轮次
        先后排列，每项为 ``id`` / ``created_at`` / ``steps``。

    Raises:
        HTTPException: 404——会话不存在。
    """
    session = await Session.get_or_none(id=session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"会话不存在：{session_id}"
        )

    turns = await Turn.filter(session_id=session_id).order_by("id")
    return {
        "id": session.id,
        "thread_id": session.thread_id,
        "title": session.title,
        "created_at": session.created_at,
        "turns": [
            {"id": turn.id, "created_at": turn.created_at, "steps": turn.steps}
            for turn in turns
        ],
    }


@router.delete("/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: int) -> None:
    """删除会话及其短期记忆。

    按会话主键删除（列表接口返回的 ``id``），短期记忆则按该会话的 ``thread_id`` 清理。
    顺序是先清记忆、再删记录：记忆清理失败时整个请求失败，会话记录还在，调用方可以重试；
    反过来先删记录的话，记忆就再也没有入口可以清理了。

    Args:
        session_id: 会话主键。

    Raises:
        HTTPException: 404——会话不存在；503——清除短期记忆失败（会话保留，可重试）。
    """
    session = await Session.get_or_none(id=session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"会话不存在：{session_id}"
        )

    # 先确认记忆清干净了再删记录：清记忆失败时直接报错，会话保持原样，调用方可以重试；
    # 反过来先删记录的话，记忆就再也没有入口可以清理了
    try:
        # 会话标识是雪花 ID，checkpointer 按字符串存取
        await retriever.forget(str(session.thread_id))
    except Exception as exc:  # noqa: BLE001 - 记忆未清除时不能删会话
        logger.opt(exception=exc).error(
            "清除会话记忆失败，会话未删除 session={} thread={}",
            session_id,
            session.thread_id,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"清除会话记忆失败，会话未删除：{exc}",
        ) from exc

    await session.delete()
    logger.info(
        "会话已删除 session={} thread={}", session_id, session.thread_id
    )
