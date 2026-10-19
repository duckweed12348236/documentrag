"""评估数据集接口：创建 / 列表 / 删除 / 评测。

四个接口都不在请求里跑重活：创建与评测只建任务记录再交给调度器
（:func:`~plugins.celery.tasks.submit`），立即返回任务字典（HTTP 202），进度从
``GET /task`` 查——状态 ``waiting`` 表示在执行队列里排队（所有类型的任务合计同时至多
:data:`~plugins.redis.TASK_MAX_CONCURRENT` 个在执行）。

1. ``POST /evaluation/dataset``：按文档 ID 用其 Markdown 生成测试用例（后台任务
   ``dataset_create``，产出见 ``outputs`` 的 ``dataset_id`` / ``items``）；
2. ``GET /evaluation/dataset``：列出数据集，可按名称模糊匹配（大小写不敏感的包含）；
3. ``DELETE /evaluation/dataset/{dataset_id}``：删除数据集（用例与报告随外键级联删除），
   并一并删除该数据集的短期记忆（PostgreSQL checkpoint + Redis 缓存）；
4. ``POST /evaluation/{dataset_id}``：对已有数据集跑评测（后台任务 ``dataset_eval``，
   产出见 ``outputs`` 的 ``result_id`` 与各指标均分 ``summary``）。

``Evaluator`` 是进程内单例插件：应用启动时由 ``register_plugins()`` 建好短期记忆，因此
本模块导入即可用（删除数据集时据此清理记忆）。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from logger import logger
from models.dataset import Dataset
from plugins.agents.evaluation.graph import Evaluator
from plugins.celery.tasks import (
    ACCEPTED_MESSAGE,
    DATASET_TASK_TYPE,
    EVALUATION_TASK_TYPE,
    submit,
)
from plugins.redis import Task, TaskStatus
from plugins.snowflake import Snowflake
from schemas.evaluation import DatasetCreate

router = APIRouter(prefix="/evaluation")
evaluator = Evaluator()


@router.post("/dataset", status_code=status.HTTP_202_ACCEPTED)
async def create_dataset(payload: DatasetCreate) -> dict:
    """提交数据集创建任务（后台生成测试用例）。

    Args:
        payload: :class:`~schemas.evaluation.DatasetCreate`——文档 ID 与用例条数；文档由其
            转换出的 Markdown 提供（入库链路的产物）。

    Returns:
        任务记录字典（``Task.model_dump()``）：状态为 ``waiting``（排队）或 ``running``
        （已投递）；生成完成后 ``outputs`` 带上 ``dataset_id`` / ``name`` / ``items``。

    Raises:
        HTTPException: 503——任务记录写入失败（Redis 不可用）或投递失败（消息代理不可用）。
    """
    return await _submit(DATASET_TASK_TYPE, {
        "document_id": payload.document_id,
        "size": payload.size,
        # 会话标识即该数据集的短期记忆线程（雪花 ID），生成时随数据集一起落库
        "thread_id": Snowflake.get_id(),
    })


@router.get("/dataset")
async def list_datasets(
        name: str | None = Query(
            None, description="按名称模糊匹配（大小写不敏感的包含）"
        ),
        limit: int | None = Query(None, ge=1, description="最多返回多少条；缺省返回全部"),
) -> list[dict]:
    """列出数据集（最新在前），可按名称模糊匹配。

    用例与评估报告随数据集一起返回（``prefetch_related`` 两条附加查询，不在循环里逐条查）：
    客户端拿到的就是「数据集 + 它的全部用例 + 历次评估结果」，不必再按 ID 逐条拉。

    Args:
        name: 名称关键字；缺省返回全部数据集。
        limit: 最多返回多少条；缺省返回全部。

    Returns:
        数据集字典列表，每项含：

        - ``id`` / ``name`` / ``thread_id``（短期记忆线程）；
        - ``items``：用例列表（按主键升序），每项 ``id`` / ``question`` / ``reference`` /
          ``reference_contexts``；
        - ``results``：评估报告列表（按主键降序，最新在前），每项 ``id`` / ``summary``
          （各指标均分）/ ``content``（大模型解读）/ ``error``（失败原因，成功为 ``None``）。
    """
    query = Dataset.all().prefetch_related("items", "results")
    if name:
        query = query.filter(name__icontains=name)
    query = query.order_by("-id")
    if limit is not None:
        query = query.limit(limit)
    datasets = await query
    return [
        {
            "id": dataset.id,
            "name": dataset.name,
            "thread_id": dataset.thread_id,
            # prefetch 回来的关联对象不保证顺序，这里显式排：用例按生成顺序、报告最新在前
            "items": [
                {
                    "id": item.id,
                    "question": item.question,
                    "reference": item.reference,
                    "reference_contexts": list(item.reference_contexts or []),
                }
                for item in sorted(dataset.items, key=lambda item: item.id)
            ],
            "results": [
                {
                    "id": result.id,
                    "summary": result.summary,
                    "content": result.content,
                    "error": result.error,
                }
                for result in sorted(
                    dataset.results, key=lambda result: result.id, reverse=True
                )
            ],
        }
        for dataset in datasets
    ]


@router.delete("/dataset/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: int) -> None:
    """删除数据集，并一并删除它的短期记忆。

    用例与报告由外键级联删除；记忆分两处（PostgreSQL checkpoint 与 Redis 缓存），统一
    交给 :meth:`~plugins.agents.evaluation.graph.Evaluator.forget` 清理——在删库之后清理，
    失败只记日志：数据集已经不在了，没有重试的落点，也不该因此返回错误。

    Args:
        dataset_id: 数据集主键。

    Raises:
        HTTPException: 404——数据集不存在。
    """
    dataset = await Dataset.get_or_none(id=dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"数据集不存在：{dataset_id}"
        )
    thread_id = dataset.thread_id
    await dataset.delete()
    logger.info("数据集已删除 dataset={} thread={}", dataset_id, thread_id)
    try:
        await evaluator.forget(thread_id)
    except Exception as exc:  # noqa: BLE001 - 数据集已删除，清理失败只记日志
        logger.opt(exception=exc).error(
            "数据集的短期记忆清理失败 dataset={} thread={}", dataset_id, thread_id
        )


@router.post(
    "/{dataset_id}", status_code=status.HTTP_202_ACCEPTED
)
async def evaluate_dataset(dataset_id: int) -> dict:
    """提交数据集评测任务（后台跑检索问答、指标打分与报告解读）。

    Args:
        dataset_id: 数据集主键。

    Returns:
        任务记录字典（``Task.model_dump()``）：状态为 ``waiting``（排队）或 ``running``
        （已投递）；评测完成后 ``outputs`` 带上 ``result_id`` 与各指标均分 ``summary``。

    Raises:
        HTTPException: 404——数据集不存在；503——任务记录写入或投递失败。
    """
    dataset = await Dataset.get_or_none(id=dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"数据集不存在：{dataset_id}"
        )
    return await _submit(EVALUATION_TASK_TYPE, {"dataset_id": dataset_id})


async def _submit(task_type: str, inputs: dict) -> dict:
    """建任务记录并交给调度器投递，返回最新的任务字典。

    Args:
        task_type: 任务类型（``dataset_create`` / ``dataset_eval``）。
        inputs: 任务的输入信息，存进任务记录的 ``inputs``。

    Returns:
        任务记录字典：执行位置已满时为 ``waiting``（排队），已投递则为 ``running``。

    Raises:
        HTTPException: 503——任务记录写入失败（Redis 不可用）或投递失败（消息代理不可用）。
    """
    task_id = Snowflake.get_id()
    try:
        task = await Task.create(
            task_id,
            task_type=task_type,
            inputs=inputs,
            message=ACCEPTED_MESSAGE,
        )
    except Exception as exc:
        logger.exception(
            "提交任务失败：任务记录写入异常 task_id={} type={}", task_id, task_type
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"任务记录写入失败（Redis 不可用）：{exc}",
        ) from exc

    try:
        await submit(task)
    except Exception as exc:
        logger.exception(
            "提交任务失败：任务投递异常 task_id={} type={}", task_id, task_type
        )
        await task.finish(status=TaskStatus.ERROR, error=f"任务投递失败：{exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"任务投递失败（消息代理不可用）：{exc}",
        ) from exc

    # 投递时任务可能刚被置为 running（调度器读的是另一份记录），回读一次给出真实状态
    return ((await Task.fetch(task_id)) or task).model_dump()
