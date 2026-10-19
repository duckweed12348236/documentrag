"""Tortoise ORM 公共模型基类。

:class:`Model` 在 Tortoise 的 :class:`~tortoise.models.Model` 之上提供
:meth:`Model.dump`：把实例连同已加载的关联对象递归展开成普通字典，便于直接序列化
（如作为接口响应）。反向关系与多对多关系须先 ``prefetch_related``，否则该字段置
``None``（未预取的关系不可同步迭代）。
"""

from typing import Any

from tortoise.exceptions import NoValuesFetched
from tortoise.fields.relational import ReverseRelation
from tortoise.models import Model as TortoiseModel


class Model(TortoiseModel):
    excluded_fields = {"_partial", "_saved_in_db", "_custom_generated_pk", "_await_when_save"}

    def dump(self) -> dict[str, Any]:
        pairs = {}
        excluded_fields = self.excluded_fields
        for key, value in self.__dict__.items():
            if key in excluded_fields:
                continue
            if key.startswith("_"):
                key = key[1:]

            if isinstance(value, Model):
                pairs[key] = value.dump()
            elif isinstance(value, ReverseRelation):
                # 反向关系（含多对多）：已 prefetch 时展开为列表；未预取时置 None——
                # 未预取的关系迭代会抛 NoValuesFetched，且原始对象不可 JSON 序列化
                try:
                    pairs[key] = [item.dump() for item in value]
                except NoValuesFetched:
                    pairs[key] = None
            elif isinstance(value, list) and value and isinstance(value[0], Model):
                pairs[key] = [item.dump() for item in value]
            else:
                pairs[key] = value
        return pairs


# 业务模型：在此导入以触发 Tortoise 注册（本模块先定义 Model，再导入子模块，子模块内
# ``from models import Model`` 不会形成循环导入）。``TORTOISE_MODELS = ["models"]``
# 只导入本包，不导入子模块，故必须显式导入。
from models.dataset import Dataset, DatasetItem, Result, ResultDetail  # noqa: E402,F401
from models.session import Session, Turn  # noqa: E402,F401
