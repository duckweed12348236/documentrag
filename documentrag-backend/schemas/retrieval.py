"""检索接口的请求模型。"""

from pydantic import BaseModel, ConfigDict, Field


class Question(BaseModel):
    """``POST /retrieval`` 的请求体。

    校验不通过由 FastAPI 统一返回 422：问题缺失、非字符串或去空白后为空即为不合法，
    因此调用方不必自己判断空问题。
    """

    #: 校验前先去掉首尾空白：纯空白的问题被 ``min_length`` 判为不合法
    model_config = ConfigDict(str_strip_whitespace=True)

    #: 本轮用户问题；多轮会话中只给本轮问题即可，上下文由 ``session_id`` 对应的短期记忆提供
    content: str = Field(..., min_length=1, description="本轮用户问题")
    #: 会话主键：继续已有会话时传它；缺省则新建会话，新会话主键由响应头 ``X-Session-Id`` 返回
    session_id: int | None = Field(
        None,
        description="会话主键；缺省时新建会话，新会话主键见响应头 X-Session-Id",
    )
