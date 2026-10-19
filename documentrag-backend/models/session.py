"""检索会话与轮次相关模型（Tortoise ORM）。

- :class:`Session`：一次多轮对话（会话），与检索流水线的短期记忆一一对应——
  :attr:`Session.thread_id`（雪花 ID）即
  :meth:`plugins.agents.retrieval.graph.Retriever.invoke` 的 ``thread_id``（checkpointer 按
  字符串存取，调用时转成字符串），同一个 ``thread_id`` 的对话状态由 checkpointer 保存与恢复；
- :class:`Turn`：会话中的一轮问答，保存该轮流水线上报的全部进度步骤。

流式写入多轮问答：:meth:`Session.open_turn` 取（或新建）会话并开一轮问答，之后每收到流水线
上报的一步就落一次库（:meth:`Turn.write`），或直接边写边透传（:meth:`Turn.write_stream`）::

    from plugins.agents.retrieval.graph import Retriever

    session, turn = await Session.open_turn(question)            # 首轮自动建会话
    async for step in turn.write_stream(
            Retriever().invoke(question, str(session.thread_id))):
        yield step                                              # 同时已写进 turn.steps

单步写入走 PostgreSQL 的 ``jsonb`` 服务端拼接（``steps || $1::jsonb``）：一条 ``UPDATE``
把新步骤追加到数组末尾，不需要把整列读回再写回，写入量只与这一步的大小相关。

这两个模型继承 :class:`models.Model`，因此可直接用 :meth:`models.Model.dump`
序列化为普通字典；会话连同其轮次需先 ``prefetch_related("turns")``，否则
``turns`` 置 ``None``。
"""

import json
from collections.abc import AsyncIterator
from dataclasses import asdict, is_dataclass
from typing import Any

from tortoise import fields

from logger import logger
from models import Model
from plugins.snowflake import Snowflake

#: 会话标题的最大长度（与 :attr:`Session.title` 字段一致）：超长的问题会被截断
TITLE_MAX_LENGTH = 255
#: 问题为空时的兜底标题
DEFAULT_SESSION_TITLE = "未命名会话"

#: 追加一步进度：由 PostgreSQL 在服务端把新步骤拼到 ``steps`` 数组末尾。
#: ``%s`` 是 psycopg 的占位符（Tortoise 的 psycopg 客户端直接把 SQL 交给游标）。
_APPEND_STEP_SQL = 'UPDATE "turn" SET "steps" = "steps" || %s::jsonb WHERE "id" = %s'


def _as_dict(step: Any) -> dict[str, Any]:
    """把一步进度转成落库用的字典。

    ``dataclass``（如 :class:`~plugins.agents.retrieval.schemas.Step`）用
    :func:`~dataclasses.asdict` 递归展开（嵌套的 chunk 等一并变成字典）；已经是映射的
    原样复制，避免后续改动影响调用方持有的对象。

    Args:
        step: 一步进度：dataclass 或映射。

    Returns:
        可 JSON 序列化的字典。
    """
    if is_dataclass(step) and not isinstance(step, type):
        return asdict(step)
    return dict(step)


class Session(Model):
    """一次多轮对话（会话）。"""

    #: 主键：雪花算法生成的 64 位 ID（默认由 :meth:`Snowflake.get_id` 生成）
    id = fields.BigIntField(pk=True, generated=False, default=Snowflake.get_id)
    #: 会话标识：检索流水线使用的 ``thread_id``（雪花 ID，checkpointer 里按字符串存取）。
    #: 一个会话对应一个线程，故唯一
    thread_id = fields.BigIntField(unique=True, generated=False)
    #: 会话标题，供会话列表展示
    title = fields.CharField(max_length=255)
    #: 创建时间（UTC，插入时由 ORM 填入）
    created_at = fields.DatetimeField(auto_now_add=True)
    #: 反向关系：该会话下的全部轮次（``Turn.session`` 的反向端）
    turns: fields.ReverseRelation["Turn"]

    class Meta:
        table = "session"

    @classmethod
    async def open_turn(
            cls, question: str, session_id: int | None = None
    ) -> tuple["Session", "Turn"]:
        """取（或新建）会话，并为这一轮问答开一条待写入的轮次。

        给出了 ``session_id`` 就复用该会话（标题不再变化）；缺省则新建会话——``thread_id``
        （检索流水线的记忆线程，雪花 ID）由服务端生成，标题取本轮问题。每轮问答都新建一条
        :class:`Turn`，``steps`` 先为空，随后由 :meth:`Turn.write` / :meth:`Turn.write_stream`
        边执行边补全。

        Args:
            question: 本轮用户问题；新建会话时用它作标题（超长截断、为空兜底）。
            session_id: 会话主键；缺省新建会话。

        Returns:
            ``(会话, 本轮轮次)``：会话供调用方取主键与 ``thread_id``（记忆线程）。

        Raises:
            DoesNotExist: ``session_id`` 指定的会话不存在。
        """
        title = (question.strip() or DEFAULT_SESSION_TITLE)[:TITLE_MAX_LENGTH]
        if session_id is None:
            session = await cls.create(thread_id=Snowflake.get_id(), title=title)
        else:
            session = await cls.get(id=session_id)
        return session, await Turn.create(session=session, steps=[])


class Turn(Model):
    """会话中的一轮问答。

    :attr:`steps` 保存该轮流水线上报的全部 :class:`~plugins.agents.retrieval.schemas.Step`
    序列化后的字典列表：节点开始为 ``running``、结束为 ``success`` / ``error``；答案生成
    步骤带上答案，重排步骤带上检索到的片段。步骤按上报顺序**逐步落库**（见
    :meth:`write`），因此对话进行到一半时也能查到已经发生的步骤。
    """

    #: 主键：雪花算法生成的 64 位 ID（默认由 :meth:`Snowflake.get_id` 生成）
    id = fields.BigIntField(pk=True, generated=False, default=Snowflake.get_id)
    #: 该轮的全部进度步骤（``Step`` 序列化后的字典列表，落库为 JSON）
    steps = fields.JSONField()
    #: 创建时间（UTC，插入时由 ORM 填入）
    created_at = fields.DatetimeField(auto_now_add=True)
    #: 所属会话；删除会话时级联删除其轮次
    session = fields.ForeignKeyField(
        "models.Session", related_name="turns", on_delete=fields.CASCADE
    )

    class Meta:
        table = "turn"

    async def write(self, step: Any) -> None:
        """把一步进度追加到本轮（流式写入的一步）。

        写入由 PostgreSQL 在服务端完成：一条 ``UPDATE`` 把新步骤拼到 ``steps`` 末尾
        （``steps || %s::jsonb``）。相比「读回整列 → 追加 → 整列写回」，这样写只发送这一步
        的数据，也不会因为并发的两次追加互相覆盖。本地对象同步追加一份，使
        ``self.steps`` 与本轮已落库的内容保持一致。

        Args:
            step: 流水线上报的一步，``dataclass``（如
                :class:`~plugins.agents.retrieval.schemas.Step`，内部字段会递归展开）或
                可直接 JSON 序列化的字典。
        """
        payload = _as_dict(step)
        await self._meta.db.execute_query(
            _APPEND_STEP_SQL,
            [json.dumps([payload], ensure_ascii=False), self.id],
        )
        self.steps = [*self.steps, payload]

    async def write_stream(self, steps: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """边写边透传：把步骤流逐条写入本轮，并把同一批步骤原样产出。

        适合直接接在接口的流式响应上：调用方 ``yield`` 出去的就是流水线的步骤，数据库里
        同时留下了这一轮的完整进度。写库失败只记日志、不打断推送——用户正在看答案，历史
        记录缺失不该让对话中断；流水线自身抛出的异常照常向外传递。

        Args:
            steps: 流水线上报的步骤流，如
                :meth:`~plugins.agents.retrieval.graph.Retriever.invoke`。

        Yields:
            与入参完全相同的步骤对象（不做任何加工）。
        """
        async for step in steps:
            try:
                await self.write(step)
            except Exception as exc:  # noqa: BLE001 - 写库失败不应打断流式推送
                logger.opt(exception=exc).error(
                    "会话轮次写入失败 turn={} session={}", self.id, self.session_id
                )
            yield step
