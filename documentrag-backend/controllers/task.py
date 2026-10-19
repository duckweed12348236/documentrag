"""任务接口：列表、重试与删除。

``GET /tasks`` 列出全部后台任务（文档入库、数据集创建、数据集评测），最新在前——任务
不只有入库一种，查询接口因此单独挂在这里，不挂在某个业务前缀下。客户端提交后直接拉列表
即可看到进度，无需先知道任务 ID；可用 ``status`` 过滤状态、``type`` 过滤任务类型、
``limit`` 限制条数。

``POST /tasks/{task_id}/retry`` 把失败的任务重新排队：只有 ``error`` 状态的任务可以重试
（其它状态返回 409），重试复用原任务的输入（``inputs``），清掉失败原因与上次的业务产出
（``outputs``），并在进度末尾追加一条「任务重新排队」。

``DELETE /tasks/{task_id}`` 删除任务记录：排队中与已结束的任务可以直接删除，执行中的任务
返回 409——删除记录并不会取消正在跑的任务，worker 收尾时还会把记录写回来。

任务记录的结构见 :class:`~plugins.redis.Task`：``created_at`` 是受理时间（排队即按它的
先后执行），``status`` 为 ``waiting`` 表示在执行队列里排队（所有类型的任务合计同时至多
:data:`~plugins.redis.TASK_MAX_CONCURRENT` 个在执行），``outputs`` 是产出（成功时的数据集
主键、报告主键等），其中 ``progress`` 键下是按顺序累积的进度步骤（末条即最新进展）。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from logger import logger
from plugins.celery.tasks import submit
from plugins.redis import Task, TaskStatus

router = APIRouter(prefix="/task")


@router.get("")
async def list_tasks(
        status_filter: TaskStatus | None = Query(
            None,
            alias="status",
            description="只看某种状态：waiting / running / success / error",
        ),
        task_type: str | None = Query(
            None,
            alias="type",
            description="只看某种任务类型：document_ingest / dataset_create / dataset_eval",
        ),
        limit: int | None = Query(None, ge=1, description="最多返回多少条；缺省返回全部"),
) -> list[dict]:
    """列出任务（普通字典，最新在前）。

    Args:
        status_filter: 只返回该状态的任务；缺省返回全部状态。
        task_type: 只返回该类型的任务；缺省返回全部类型。
        limit: 最多返回多少条；缺省返回全部。

    Returns:
        任务记录字典列表（``Task.model_dump()``），最新在前。
    """
    tasks = await Task.list(status=status_filter, task_type=task_type, limit=limit)
    return [task.model_dump() for task in tasks]


@router.post("/{task_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_task(task_id: int) -> dict:
    """重试一个失败的任务：重新排队并交给调度器投递。

    重试不是就地续跑：任务状态回到 ``waiting``，由调度器按创建时间重新排队，执行时任务体
    从 ``inputs`` 重新跑一遍（上一次的进度保留在 ``outputs.progress`` 里，便于对照失败到
    哪一步）。

    Args:
        task_id: 任务 ID。

    Returns:
        任务记录字典（``Task.model_dump()``）：状态为 ``waiting``（排队中，执行位置已满）或
        ``running``（已重新投递）；``error`` 已清空、``outputs`` 只剩进度。

    Raises:
        HTTPException: 404——任务不存在；409——任务不是 ``error`` 状态（只有失败的任务可以
            重试）；503——任务投递失败（消息代理不可用）。
    """
    task = await Task.fetch(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"任务不存在：{task_id}"
        )
    if task.status != TaskStatus.ERROR:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"只有失败的任务可以重试：当前状态为 {task.status.value}",
        )

    await task.retry()
    logger.info("任务重新排队 task_id={} type={}", task.id, task.type)
    try:
        await submit(task)
    except Exception as exc:
        logger.exception("重试任务失败：任务投递异常 task_id={}", task_id)
        await task.finish(status=TaskStatus.ERROR, error=f"任务投递失败：{exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"任务投递失败（消息代理不可用）：{exc}",
        ) from exc

    # 投递时任务可能刚被置为 running（调度器读的是另一份记录），回读一次给出真实状态
    return ((await Task.fetch(task_id)) or task).model_dump()


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(task_id: int) -> None:
    """删除一个任务记录。

    任务记录只是一份状态快照：删掉它既不会取消已经在跑的任务，也不会阻止 worker 把结果写
    回来，因此**执行中**（``running``）的任务拒绝删除；排队中与已结束（``waiting`` /
    ``success`` / ``error``）的任务直接删除。

    Args:
        task_id: 任务 ID。

    Raises:
        HTTPException: 404——任务不存在；409——任务正在执行（不能删除）。
    """
    task = await Task.fetch(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"任务不存在：{task_id}"
        )
    if task.status is TaskStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"执行中的任务不能删除：{task_id}",
        )
    await task.remove()
    logger.info(
        "任务已删除 task_id={} type={} status={}",
        task.id, task.type, task.status.value,
    )
