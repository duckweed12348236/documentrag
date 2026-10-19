"""评估流水线的节点实现（数据集生成 / 数据集评测）。

两个节点首尾相接、永远循环：评测完回到生成节点，生成节点再把它送回评测节点；一轮一轮地
转，靠评测节点前的断点每次 ``invoke`` 只放行一轮。

:func:`generate_dataset`（测试用例生成 / 路由）：

1. ``state.dataset_id`` 非空（数据集已经存在）：**只做路由**——直接跳转到评测节点
   （:data:`EVAL_DATASET_NODE`），不读库、不重新生成，因此不会重复生成测试数据集；
2. 否则用状态里的文档片段（``state.content``：由
   :meth:`~plugins.agents.evaluation.graph.Evaluator.invoke` 读盘并按标题切章节、按 token
   切小块后写入的正文文本，与入库流水线一致），在节点内组装成文档对象后交给生成器；
   没有文档片段且 ``dataset_id`` 为空时跳 ``END``；
3. 从运行时上下文取出生成器（``runtime.context.generator``，见
   :class:`~plugins.agents.evaluation.schemas.Context`），把全部文档交给它，按
   ``state.size`` 生成测试用例——生成器由调用方构造，内部已包装好大模型与向量模型
   （如 :data:`~plugins.agents.models.ds` 与
   :class:`~plugins.agents.models.QwenEmbeddingModel`）。Ragas 只提供同步生成接口，
   这里用 :func:`asyncio.to_thread` 放到线程中执行，避免阻塞事件循环；
4. 每条用例转换成 :class:`~plugins.agents.evaluation.schemas.DatasetItemInfo`
   （``question`` 取 Ragas 的 ``user_input``、``reference`` 取参考答案、
   ``reference_contexts`` 取参考上下文）；
5. 在同一事务内把数据集与用例写入数据库：创建
   :class:`~models.dataset.Dataset`（名称取 ``数据集-<雪花ID>``，``thread_id`` 取运行时
   给出的会话标识——即本次运行的短期记忆线程，缺省兜底雪花 ID），再批量创建
   :class:`~models.dataset.DatasetItem`；随后回传 ``dataset_id``（数据集主键）并跳到评测
   节点（断点就在那里，生成任务到这一步就停）。

容错：``dataset_id`` 与 ``content`` 同时为空、运行时上下文未注入生成器、生成失败或落库
失败时记录日志，不写状态、跳 ``END`` 结束流水线（文档读不出来由
:meth:`~plugins.agents.evaluation.graph.Evaluator.invoke` 在切分时上报失败步骤）。

:func:`eval_dataset`（数据集评测）：

1. 按 ``state.dataset_id`` 从数据库读出该数据集及其用例（数据集不存在、没有可用用例时上报
   原因并跳 ``END``）——用例列表只在本次节点内使用，不写进状态；
2. 以 :func:`asyncio.gather` **并发**调用运行时上下文的检索智能体
   （``runtime.context.retriever.eval``，见
   :class:`~plugins.agents.retrieval.graph.Retriever`），传入用例 ``question`` 与
   条目 key（该条目在 ``dataset_infos`` 中的下标，用于把结果定位回条目），取回检索生成的
   回答与检索到的上下文；单条失败只告警、不影响其余用例；
3. 用检索结果里的条目 key 定位回对应数据集条目，逐个收集为
   :attr:`~plugins.agents.evaluation.schemas.ResultInfo.details`；
4. 每个指标一次 ``abatch_score`` 批量打分（内部并发全部用例；指标取自运行时上下文的
   ``context.metrics``，见 :class:`~plugins.agents.evaluation.schemas.Context`），分项得分
   写回对应条目（:attr:`~plugins.agents.evaluation.schemas.ResultDetailInfo.scores`），并
   汇总各指标的整体平均分；
5. 把各项用例情况与整体均分按
   :data:`~plugins.agents.evaluation.prompts.REPORT_INTERPRETATION_PROMPT` 约定的 JSON
   结构提交运行时上下文里的大模型（``runtime.context.model``）解读，调用走
   :func:`~plugins.agents.models.invoke`（输出用
   :class:`~plugins.agents.models.TextOutputValidator` 只校验非空），报告正文落到
   :attr:`~plugins.agents.evaluation.schemas.ResultInfo.content`；
6. 在同一事务内把报告写入数据库：先清掉该数据集已有的报告（同一数据集重新评测即覆盖旧
   报告），再创建 :class:`~models.dataset.Result`（关联 ``state.dataset_id``）与批量创建
   :class:`~models.dataset.ResultDetail`；最后以 ``result_info`` 回传并**跳回生成节点**，
   开始下一轮。

重复评测：评测节点每次都跳回生成节点——此时 ``dataset_id`` 已非空，生成节点只做路由、立刻
跳回评测节点，断点设在评测节点之前，所以本次运行会停在那里，**一次
:meth:`~plugins.agents.evaluation.graph.Evaluator.invoke` 只多评测一轮**，可以一直评下去。
每轮用例都重新从库里读、每条报告都覆盖旧的，``dataset_id`` 已在状态里，不会重复生成数据集。
会话的长期占用由调用方按需清理：
:meth:`~plugins.agents.evaluation.graph.Evaluator.forget` 或删除数据集。

容错：状态缺少 ``dataset_id``、数据集不存在、没有待评测用例时记录日志、跳 ``END``；其余
失败（全部检索失败、全部指标打分失败、报告解读失败、落库失败）除记录日志外，还会把原因
写入 :attr:`~models.dataset.Result.error`（覆盖该数据集的旧报告），随后与成功时一样跳回
生成节点——因此失败后再调一次 ``invoke`` 就是一次重试。

进度：两个节点都通过 ``runtime.stream_writer`` 上报
:class:`~plugins.agents.evaluation.schemas.Step`——每个步骤开始时为 ``running``、结束为
``success``；跳过与失败时为 ``error`` 并带上原因（跳过同样算一次被上报的步骤）。生成节点
的步骤为「正在生成测试用例」「正在保存数据集」（``dataset_id`` 非空时只做跳转、不上报步骤），
切分文档的「正在读取文档」由 :meth:`~plugins.agents.evaluation.graph.Evaluator.invoke` 上报；
评测节点的步骤为「正在读取数据集」「正在检索用例」「正在评测指标」「正在解读评测报告」
「正在保存评测报告」，前置条件不满足时以「正在评测数据集」上报原因。
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import asdict
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command
from tortoise.transactions import in_transaction

from logger import logger
from models.dataset import Dataset, DatasetItem, Result, ResultDetail
from plugins.agents.evaluation.prompts import REPORT_INTERPRETATION_PROMPT
from plugins.agents.evaluation.schemas import (
    Context,
    DatasetItemInfo,
    ResultDetailInfo,
    ResultInfo,
    State,
    Step,
    StepStatus,
)
from plugins.agents.models import TextOutputValidator, invoke
from plugins.snowflake import Snowflake


#: 数据集评测节点的名字（图上的节点名）：已有数据集时由生成节点直接跳转过去
EVAL_DATASET_NODE = "eval_dataset"

#: 数据集生成节点的名字（图上的节点名）：评测完跳回来，开始下一轮
GENERATE_DATASET_NODE = "generate_dataset"


class EvaluationError(RuntimeError):
    """评测中途失败：错误信息会写入 :attr:`~models.dataset.Result.error`。"""


async def _score_all(
        metrics: list[Any], cases: list[tuple[DatasetItemInfo, ResultDetailInfo]]
) -> dict[str, list[float]]:
    """对全部指标并发批量打分，返回 ``{指标名: [各用例得分]}``。

    各指标所需输入不同（如 AnswerRelevancy 要 ``user_input`` + ``response``，ContextRecall
    还要 ``retrieved_contexts`` + ``reference``），按 ``ascore`` 的形参过滤后交给
    :meth:`~ragas.metrics.base.BaseMetric.abatch_score`——它内部并发全部用例、保序返回，因此
    得分与 ``cases`` 一一对应。单个指标整批失败（内部不隔离异常）时只告警并跳过该指标，
    不影响其余指标；返回为空表示全部指标都失败。
    """
    async def score(metric: Any) -> list[float]:
        parameters = inspect.signature(metric.ascore).parameters
        scored = await metric.abatch_score([
            {
                name: value
                for name, value in (
                    ("user_input", dataset_info.question),
                    ("response", case.answer),
                    ("retrieved_contexts", case.retrieved_contexts),
                    ("reference", dataset_info.reference),
                )
                if name in parameters
            }
            for dataset_info, case in cases
        ])
        return [float(result.value) for result in scored]

    groups = await asyncio.gather(
        *(score(metric) for metric in metrics), return_exceptions=True
    )
    results = {}
    for metric, scores in zip(metrics, groups):
        if isinstance(scores, BaseException):
            logger.opt(exception=scores).error(
                "指标评测失败 metric={} cases={}", metric.name, len(cases)
            )
            continue
        results[metric.name] = scores
    return results


async def generate_dataset(state: State, runtime: Runtime[Context]) -> Command:
    """生成测试用例；数据集已存在时只做路由，直接转去评测。

    两条互斥的入口路径：

    - ``state.dataset_id`` 非空：**只判断、不读写任何东西**——直接跳转到评测节点
      （:data:`EVAL_DATASET_NODE`）。数据集与用例由评测节点按主键自己读取，所以循环评测时
      本节点每轮只做一次路由，绝不会有第二次生成；
    - 否则用状态里的文档片段（``state.content``：由
      :meth:`~plugins.agents.evaluation.graph.Evaluator.invoke` 读盘并按标题 / token 切分好
      的正文文本，与入库流水线一致）在节点内**组装成文档对象**，生成测试用例并落库为新
      数据集，回传 ``dataset_id`` 后同样跳转到评测节点（:data:`EVAL_DATASET_NODE`）。

    两者都为空时跳 ``END``。

    生成路径的细节：生成器从运行时上下文取（``runtime.context.generator``，见
    :class:`~plugins.agents.evaluation.schemas.Context`），由调用方构造并注入，内部已包装
    好大模型与向量模型；用例条数为 ``state.size``。由于 Ragas 只提供同步的
    :meth:`~ragas.testset.TestsetGenerator.generate_with_langchain_docs`，生成用
    :func:`asyncio.to_thread` 丢到线程中执行，期间不阻塞事件循环。生成结果的每条样本
    （``TestsetSample.eval_sample``）取出 ``user_input`` / ``reference`` /
    ``reference_contexts`` 转成 :class:`~plugins.agents.evaluation.schemas.DatasetItemInfo`，
    随后在同一事务内把 :class:`~models.dataset.Dataset`（名称取 ``数据集-<雪花ID>``）与
    对应的 :class:`~models.dataset.DatasetItem` 写入数据库。

    Args:
        state: 流水线状态——``dataset_id``（非空时只做路由、不再生成）、``content``
            （切分好的文档片段，只在 ``dataset_id`` 为空时使用，由本节点组装成文档对象）与
            ``size``（生成用例条数）。
        runtime: LangGraph 运行时——``context.generator`` 取测试用例生成器、
            ``execution_info.thread_id`` 取本次运行的会话标识（随新建的数据集一起落库）；
            步骤进度由 ``stream_writer`` 上报。

    Returns:
        ``Command``：``dataset_id`` 非空时直接 ``goto`` 评测节点（:data:`EVAL_DATASET_NODE`，
        不改状态）；有文档片段时把数据集与用例写入数据库
        （:class:`~models.dataset.Dataset` / :class:`~models.dataset.DatasetItem`），以
        ``dataset_id`` 回传并跳到评测节点（断点就在那里，生成任务到这一步就停）。``dataset_id``
        与 ``content`` 同时为空、上下文未注入生成器、生成失败或落库失败时跳 ``END``（不修改
        状态）。
    """
    if state.dataset_id is not None:
        # 数据集已经存在：只做路由，绝不重新生成
        return Command(goto=EVAL_DATASET_NODE)

    writer = runtime.stream_writer
    # 状态里只存切分好的正文，文档对象在本节点组装
    documents = [Document(page_content=piece) for piece in state.content or []]
    if not documents:
        logger.error("测试用例生成跳过：既没有待生成用例的文档，也没有 dataset_id")
        writer(Step(
            message="正在生成测试用例",
            status=StepStatus.ERROR,
            error="没有待生成用例的文档（请提供 Markdown 文档路径或已有数据集）",
        ))
        return Command(goto=END)

    logger.info("测试用例生成开始 documents={} size={}", len(documents), state.size)
    writer(Step(message="正在生成测试用例"))
    try:
        testset = await asyncio.to_thread(
            runtime.context.generator.generate_with_langchain_docs, documents, state.size
        )
    except Exception as exc:
        logger.opt(exception=exc).error("测试用例生成失败 size={}", state.size)
        writer(Step(message="正在生成测试用例", status=StepStatus.ERROR, error=str(exc)))
        return Command(goto=END)

    dataset_infos = [
        DatasetItemInfo(
            question=sample.eval_sample.user_input or "",
            reference=sample.eval_sample.reference or "",
            reference_contexts=list(sample.eval_sample.reference_contexts or []),
        )
        for sample in testset.samples
    ]
    logger.info("测试用例生成完成 dataset_infos={}", len(dataset_infos))
    writer(Step(message="正在生成测试用例", status=StepStatus.SUCCESS))

    # 落库：先建数据集，再批量写入用例；同一事务保证不会留下“有数据集无用例”的半成品
    writer(Step(message="正在保存数据集"))
    # 会话标识取自运行时（checkpointer 的 thread_id，见 ExecutionInfo.thread_id）；
    # 未启用短期记忆或不是数字（写不进 bigint 列）时兜底雪花 ID
    execution_info = runtime.execution_info
    thread_id = execution_info.thread_id if execution_info is not None else None
    try:
        async with in_transaction():
            # 名称含雪花 ID：先取 ID 再落库，省去二次更新名称
            dataset_id = Snowflake.get_id()
            dataset = await Dataset.create(
                id=dataset_id,
                name=f"数据集-{dataset_id}",
                thread_id=int(thread_id) if thread_id and thread_id.isdigit() else Snowflake.get_id(),
            )
            if dataset_infos:
                await DatasetItem.bulk_create([
                    DatasetItem(
                        dataset=dataset,
                        question=dataset_info.question,
                        reference=dataset_info.reference,
                        reference_contexts=dataset_info.reference_contexts,
                    )
                    for dataset_info in dataset_infos
                ])
    except Exception as exc:
        logger.opt(exception=exc).error("数据集保存失败")
        writer(Step(message="正在保存数据集", status=StepStatus.ERROR, error=str(exc)))
        return Command(goto=END)

    logger.info("数据集已保存 dataset={} items={}", dataset.id, len(dataset_infos))
    writer(Step(message="正在保存数据集", status=StepStatus.SUCCESS))
    # 只回传主键：用例由评测节点自己按主键读，状态里不再带用例列表。
    # 生成完交给评测节点（断点在那里，本轮的生成任务到这里就停）
    return Command(update={"dataset_id": dataset.id}, goto=EVAL_DATASET_NODE)


async def eval_dataset(state: State, runtime: Runtime[Context]) -> Command:
    """读取数据集的用例并逐条评测，再用 Ragas 现代指标打分。

    用例由本节点按 ``state.dataset_id`` 从数据库读取（:class:`~models.dataset.Dataset` 与
    :class:`~models.dataset.DatasetItem`）——用例列表只在本次节点内使用，不写进状态，所以
    循环评测的每一轮都拿到库里最新的用例；数据集不存在、没有可用用例时上报原因并跳 ``END``。

    每条用例把 ``question`` 与条目 key（该条目在用例列表中的下标，用于把结果定位回对应条目）
    交给运行时上下文的检索智能体（``runtime.context.retriever.eval``，见
    :class:`~plugins.agents.retrieval.graph.Retriever`）执行一次检索，取回
    ``(回答, 检索到的上下文列表, key)``。全部用例以 :func:`asyncio.gather` **并发**发起，
    单条失败只告警、不影响其余用例。

    检索结果用 key 定位回对应数据集条目后逐个收集；随后**每个指标一次**
    :meth:`~ragas.metrics.base.BaseMetric.abatch_score` 批量打分（内部并发全部用例；指标
    取自 ``context.metrics``，见
    :class:`~plugins.agents.evaluation.schemas.Context`），各条目的分项得分写回对应结果
    （:attr:`~plugins.agents.evaluation.schemas.ResultDetailInfo.scores`），并汇总出各指标
    的整体平均分。

    最后把各项用例情况与整体指标均分按
    :data:`~plugins.agents.evaluation.prompts.REPORT_INTERPRETATION_PROMPT` 约定的 JSON
    结构提交运行时上下文里的大模型（``runtime.context.model``）解读，报告正文落到
    :attr:`~plugins.agents.evaluation.schemas.ResultInfo.content`；随后在同一事务内把报告
    写入数据库（:class:`~models.dataset.Result` + :class:`~models.dataset.ResultDetail`，
    关联 ``state.dataset_id``）——该数据集已有报告时先清掉旧报告（重新评测即覆盖），再写入
    本次报告，并以 ``result_info`` 回传。

    跳转：这一轮评测结束回到生成节点（:data:`GENERATE_DATASET_NODE`）。那时 ``dataset_id``
    已非空，生成节点只做路由、立刻跳回本节点；断点设在评测节点之前，所以本次运行会停在那里，
    **一次 :meth:`~plugins.agents.evaluation.graph.Evaluator.invoke` 只多评测一轮**，想评
    多少轮就调多少次，每轮用例都重新从库里读，每次报告都覆盖上一次的。

    Args:
        state: 流水线状态，使用其中的 ``dataset_id``（数据集主键：用例按它读取，评测报告据此
            关联数据集）。
        runtime: LangGraph 运行时——``context.retriever`` 取检索智能体、
            ``context.metrics`` 取评测指标、``context.model`` 取解读报告的大模型；步骤进度
            由 ``stream_writer`` 上报。

    Returns:
        ``Command``：把报告写入数据库（:class:`~models.dataset.Result` /
        :class:`~models.dataset.ResultDetail`），并以 ``result_info`` 回传
        （:class:`~plugins.agents.evaluation.schemas.ResultInfo` 的 ``details`` 为各项用例
        情况、``summary`` 为整体指标均分、``content`` 为大模型的解读报告），随后跳回生成
        节点（:data:`GENERATE_DATASET_NODE`）；状态缺少 ``dataset_id``、数据集不存在、
        没有可用用例时记录日志并跳 ``END``；其余失败（全部检索失败、全部指标打分失败、报告
        解读失败、落库失败）除记录日志外，还会把原因写入
        :attr:`~models.dataset.Result.error`（覆盖该数据集的旧报告），随后同样跳回生成节点
        ——失败后再调一次 ``invoke`` 即可重试一轮。
    """
    writer = runtime.stream_writer

    dataset_id = state.dataset_id
    if dataset_id is None:
        logger.error("数据集评测跳过：状态里缺少 dataset_id，既读不到用例也无处关联报告")
        writer(Step(
            message="正在评测数据集",
            status=StepStatus.ERROR,
            error="状态里缺少 dataset_id，既读不到用例也无处关联报告",
        ))
        return Command(goto=END)

    # 用例在这里读：dataset_id 是唯一事实来源，状态里不冗余保存用例列表
    # 当前步骤的说明：开始 / 成功 / 失败都上报同一个 message，失败时据此定位到失败的步骤
    message = "正在读取数据集"
    writer(Step(message=message))
    try:
        dataset = await Dataset.get_or_none(id=dataset_id)
        dataset_infos = [] if dataset is None else [
            DatasetItemInfo(
                question=item.question,
                reference=item.reference,
                reference_contexts=list(item.reference_contexts or []),
            )
            for item in await DatasetItem.filter(dataset_id=dataset_id)
        ]
    except Exception as exc:
        logger.opt(exception=exc).error("数据集读取失败 dataset={}", dataset_id)
        writer(Step(message=message, status=StepStatus.ERROR, error=str(exc)))
        return Command(goto=END)

    if dataset is None:
        logger.error("数据集评测跳过：数据集不存在 dataset={}", dataset_id)
        writer(Step(
            message=message,
            status=StepStatus.ERROR,
            error=f"数据集不存在：{dataset_id}",
        ))
        return Command(goto=END)
    if not dataset_infos:
        logger.error("数据集评测跳过：数据集没有可用的用例 dataset={}", dataset_id)
        writer(Step(message=message, status=StepStatus.ERROR, error="数据集没有可用的用例"))
        return Command(goto=END)

    logger.info(
        "数据集用例已读取 dataset={} name={} items={}",
        dataset_id, dataset.name, len(dataset_infos),
    )
    writer(Step(message=message, status=StepStatus.SUCCESS))

    context = runtime.context
    retriever = context.retriever
    metrics = context.metrics
    model = context.model

    # 当前步骤的说明：开始 / 成功 / 失败都上报同一个 message，失败时据此定位到失败的步骤
    message = "正在检索用例"
    # 失败原因统一写入 Result.error（见 _save_result_error），这里兜底捕获全部失败路径
    try:
        logger.info("数据集评测开始 cases={}", len(dataset_infos))
        writer(Step(message=message))
        # 每条用例一次检索问答，全部并发发起；单条失败只告警，不影响其余用例
        # key 取条目在 dataset_infos 中的下标，随结果回传后用于定位回对应数据集条目
        retrievals = await asyncio.gather(
            *(retriever.eval(dataset_info.question, key) for key, dataset_info in enumerate(dataset_infos)),
            return_exceptions=True,
        )

        # 用结果里的 key 定位回数据集条目，逐个收集为 details（检索失败的条目不入其中）
        result_details = []
        cases = []
        for index, retrieval in enumerate(retrievals):
            if isinstance(retrieval, BaseException):
                logger.opt(exception=retrieval).error(
                    "数据集检索失败 question={}", dataset_infos[index].question
                )
                continue
            answer, retrieved_chunks, key = retrieval
            dataset_info = dataset_infos[key]
            # 检索结果是 Chunk 对象；指标（ragas 要 str）与落库（JSON 数组）都只认正文，
            # 这里统一取 chunk.content，别再往后面传对象
            detail = ResultDetailInfo(
                question=dataset_info.question,
                answer=answer,
                retrieved_contexts=[chunk.content for chunk in retrieved_chunks or []],
            )
            result_details.append(detail)
            cases.append((dataset_info, detail))
            logger.info("数据集检索结果 key={} question={}", key, dataset_info.question)

        if not cases:
            raise EvaluationError("全部用例检索失败，没有可用的检索结果")
        writer(Step(message=message, status=StepStatus.SUCCESS))

        # 每个指标一次批量打分（abatch_score 内部并发全部用例；整批失败的指标跳过）
        message = "正在评测指标"
        writer(Step(message=message))
        score_groups = await _score_all(metrics, cases)
        if not score_groups:
            raise EvaluationError("全部指标批量打分失败，没有产出任何得分")

        # 分项得分写回对应用例，并汇总各指标的整体平均分
        for index, (_dataset_info, case) in enumerate(cases):
            case.scores = {
                name: scores[index]
                for name, scores in score_groups.items()
                if index < len(scores)
            }
        summary = {
            name: sum(scores) / len(scores)
            for name, scores in score_groups.items()
            if scores
        }
        if not summary:
            raise EvaluationError("全部指标批量打分失败，没有产出任何得分")
        writer(Step(message=message, status=StepStatus.SUCCESS))

        # 报告解读：把各项用例情况与整体均分交给大模型，输出报告正文；
        # 调用走 invoke（输出用 TextOutputValidator 只校验非空），失败由它带上错误信息重试
        message = "正在解读评测报告"
        writer(Step(message=message))
        payload = {"details": [asdict(detail) for detail in result_details], "summary": summary}
        content, error = await invoke(
            model,
            [
                SystemMessage(content=REPORT_INTERPRETATION_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ],
            TextOutputValidator,
        )
        if error:
            raise EvaluationError(f"报告解读失败：{error}")
        writer(Step(message=message, status=StepStatus.SUCCESS))

        content = content.strip()
        result_info = ResultInfo(details=result_details, summary=summary, content=content)

        # 落库：先建报告，再批量写入逐条用例情况；同一事务保证不会留下“有报告无明细”的半成品
        message = "正在保存评测报告"
        writer(Step(message=message))
        async with in_transaction():
            # 同一数据集重新评测时覆盖旧报告：先清掉该数据集已有的报告（明细随外键级联删除）
            await Result.filter(dataset_id=dataset_id).delete()
            result = await Result.create(
                dataset_id=dataset_id, summary=summary, content=content
            )
            if result_details:
                await ResultDetail.bulk_create([
                    ResultDetail(
                        result=result,
                        question=detail.question,
                        answer=detail.answer,
                        retrieved_contexts=detail.retrieved_contexts,
                        scores=detail.scores,
                    )
                    for detail in result_details
                ])
    except Exception as exc:
        logger.opt(exception=exc).error("数据集评测失败 dataset={}", dataset_id)
        writer(Step(message=message, status=StepStatus.ERROR, error=str(exc)))
        # 失败原因写入该数据集的报告（覆盖旧报告），再调一次 invoke 就是一次重试
        try:
            async with in_transaction():
                await Result.filter(dataset_id=dataset_id).delete()
                await Result.create(
                    dataset_id=dataset_id, summary={}, content="", error=str(exc)
                )
        except Exception as save_exc:  # noqa: BLE001 - 连错误都存不进去时只记日志
            logger.opt(exception=save_exc).error(
                "评测失败原因保存失败 dataset={}", dataset_id
            )
        # 失败也回到生成节点：下一次 invoke 就是一次重试
        return Command(goto=GENERATE_DATASET_NODE)

    logger.info(
        "数据集评测完成 result={} cases={} summary={} content_chars={}",
        result.id, len(result_details), summary, len(content),
    )
    writer(Step(message=message, status=StepStatus.SUCCESS))
    # 回到生成节点接着下一轮；断点会在评测节点前把它拦下，于是本次 invoke 到此结束
    return Command(update={"result_info": result_info}, goto=GENERATE_DATASET_NODE)
