"""评估数据集相关模型（Tortoise ORM）。

- :class:`Dataset`：评估数据集，聚合若干测试用例；
- :class:`DatasetItem`：数据集中的单条测试用例——问题、参考答案与参考上下文，
  字段与 :class:`plugins.agents.evaluation.schemas.DatasetItemInfo` 对应；
- :class:`Result`：数据集的评测报告——整体指标均分与大模型解读；
- :class:`ResultDetail`：报告中的单条用例情况——回答、检索到的上下文与分项得分，
  字段与 :class:`plugins.agents.evaluation.schemas.ResultDetailInfo` 对应。

这些模型继承 :class:`models.Model`，因此可直接用 :meth:`models.Model.dump` 序列化为
普通字典（含关联对象的递归展开）。
"""

from tortoise import fields

from models import Model
from plugins.snowflake import Snowflake


class Dataset(Model):
    """评估数据集。"""

    #: 主键：雪花算法生成的 64 位 ID（默认由 :meth:`Snowflake.get_id` 生成）
    id = fields.BigIntField(pk=True, generated=False, default=Snowflake.get_id)
    name = fields.CharField(max_length=255)
    #: 短期记忆的会话标识（checkpointer 的 ``thread_id``）：一个数据集对应一条记忆线程
    thread_id = fields.BigIntField(pk=False, generated=False)
    #: 反向关系：该数据集下的全部测试用例（``DatasetItem.dataset`` 的反向端）
    items: fields.ReverseRelation["DatasetItem"]
    #: 反向关系：该数据集下的全部评测报告（``Result.dataset`` 的反向端）
    results: fields.ReverseRelation["Result"]

    class Meta:
        table = "dataset"


class DatasetItem(Model):
    """数据集中的单条测试用例。"""

    #: 主键：雪花算法生成的 64 位 ID（默认由 :meth:`Snowflake.get_id` 生成）
    id = fields.BigIntField(pk=True, generated=False, default=Snowflake.get_id)
    question = fields.TextField()
    reference = fields.TextField()
    #: 参考上下文列表（字符串数组，落库为 JSON）
    reference_contexts = fields.JSONField()
    #: 所属数据集；删除数据集时级联删除其测试用例
    dataset = fields.ForeignKeyField(
        "models.Dataset", related_name="items", on_delete=fields.CASCADE
    )

    class Meta:
        table = "dataset_item"


class Result(Model):
    """数据集评测报告：一次评测的整体汇总与大模型解读。"""

    #: 主键：雪花算法生成的 64 位 ID（默认由 :meth:`Snowflake.get_id` 生成）
    id = fields.BigIntField(pk=True, generated=False, default=Snowflake.get_id)
    #: 评测的总体情况：Ragas 指标名 → 全部条目的平均分（落库为 JSON）
    summary = fields.JSONField()
    #: 大模型对本次评测的解读（Markdown 报告正文）
    content = fields.TextField()
    #: 评估失败时的错误信息；成功为 None
    error = fields.TextField(null=True)
    #: 所属数据集；删除数据集时级联删除其评测报告
    dataset = fields.ForeignKeyField(
        "models.Dataset", related_name="results", on_delete=fields.CASCADE
    )
    #: 反向关系：报告下的逐条用例情况（``ResultDetail.result`` 的反向端）
    items: fields.ReverseRelation["ResultDetail"]

    class Meta:
        table = "result"


class ResultDetail(Model):
    """评测报告中的单条用例情况。"""

    #: 主键：雪花算法生成的 64 位 ID（默认由 :meth:`Snowflake.get_id` 生成）
    id = fields.BigIntField(pk=True, generated=False, default=Snowflake.get_id)
    question = fields.TextField()
    answer = fields.TextField()
    #: 该条目检索到的上下文列表（字符串数组，落库为 JSON）
    retrieved_contexts = fields.JSONField()
    #: 该条目的分项得分：Ragas 指标名 → 得分（落库为 JSON）
    scores = fields.JSONField()
    #: 所属报告；删除报告时级联删除其明细
    result = fields.ForeignKeyField(
        "models.Result", related_name="items", on_delete=fields.CASCADE
    )

    class Meta:
        table = "result_detail"
