"""评估接口的请求模型。"""

from pydantic import BaseModel, ConfigDict, Field

#: 生成测试用例的默认条数
DEFAULT_DATASET_SIZE = 10


class DatasetCreate(BaseModel):
    """``POST /evaluation/dataset`` 的请求体：按文档 ID 生成数据集。

    只传文档 ID，不再传正文：文档正文由入库链路解析成 Markdown 后落在
    ``storage/<文档 ID>/``，此处从对象存储的元数据里取它的路径，由评估智能体自己读取并
    切分。校验不通过由 FastAPI 统一返回 422（文档 ID 缺失或去空白后为空）。
    """

    #: 校验前先去掉首尾空白：纯空白的文档 ID 被 ``min_length`` 判为不合法
    model_config = ConfigDict(str_strip_whitespace=True)

    #: 文档 ID（提交入库时生成的 UUID，对应 ``storage/<文档 ID>/`` 与对象元数据的 ``id``）
    document_id: str = Field(..., min_length=1, description="文档 ID（UUID）")
    #: 生成的测试用例条数
    size: int = Field(DEFAULT_DATASET_SIZE, ge=1, description="生成的测试用例条数")
