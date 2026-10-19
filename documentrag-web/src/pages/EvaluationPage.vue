<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import AppIcon from '@/components/AppIcon.vue'
import {
  createDataset,
  deleteDataset,
  evaluateDataset,
  fetchDatasets,
} from '@/api/evaluation'
import type {
  Dataset,
  DatasetEvalOutput,
  DatasetResult,
  IngestInput,
  Task,
} from '@/api/types'
import { formatDateTime, formatScore, messageOf } from '@/utils/format'
import {
  TASK_STATUS_LABELS,
  documentIdOf,
  latestProgress,
  readTaskInput,
  readTaskOutput,
  taskStatusTagType,
} from '@/utils/task'
import { useTaskStore } from '@/stores/task'

interface MetricEntry {
  name: string
  score: number
}

interface DocumentOption {
  value: string
  label: string
}

const taskStore = useTaskStore()

const datasets = ref<Dataset[]>([])
const loading = ref(false)
const keyword = ref('')

const createVisible = ref(false)
const creating = ref(false)
const form = ref({ documentId: '', size: 10 })

const evaluatingId = ref<string | null>(null)

const reportVisible = ref(false)
const reportDataset = ref<Dataset | null>(null)

const totalItems = computed(() =>
  datasets.value.reduce((sum, dataset) => sum + dataset.items.length, 0),
)
const totalResults = computed(() =>
  datasets.value.reduce((sum, dataset) => sum + dataset.results.length, 0),
)
const activeEvaluationCount = computed(
  () =>
    taskStore.tasks.filter(
      (task) =>
        task.type === 'dataset_eval' &&
        (task.status === 'waiting' || task.status === 'running'),
    ).length,
)

const documentOptions = computed<DocumentOption[]>(() => {
  const map = new Map<string, DocumentOption>()
  for (const task of taskStore.tasks) {
    if (task.type !== 'document_ingest' || task.status !== 'success') continue
    const documentId = documentIdOf(task)
    if (!documentId || map.has(documentId)) continue
    const input = readTaskInput<IngestInput>(task)
    map.set(documentId, {
      value: documentId,
      label: `${input?.filename || documentId} · ${formatDateTime(task.created_at)}`,
    })
  }
  return Array.from(map.values())
})

const latestEvalTasks = computed(() => {
  const map = new Map<string, Task>()
  for (const task of taskStore.tasks) {
    if (task.type !== 'dataset_eval') continue
    const datasetId = datasetIdOfTask(task)
    if (!datasetId || map.has(datasetId)) continue
    map.set(datasetId, task)
  }
  return map
})

const reportResult = computed<DatasetResult | null>(() => {
  const results = reportDataset.value?.results
  if (!results || results.length === 0) return null
  return results[0] ?? null
})

const reportTask = computed<Task | undefined>(() => {
  const dataset = reportDataset.value
  if (!dataset) return undefined
  return latestEvalOf(dataset)
})

const reportMetrics = computed<MetricEntry[]>(() => {
  const summary = reportResult.value?.summary
  if (!summary) return []
  return Object.entries(summary)
    .map(([name, score]) => ({ name, score }))
    .sort((a, b) => a.name.localeCompare(b.name))
})

function datasetIdOfTask(task: Task): string | null {
  const output = readTaskOutput<DatasetEvalOutput>(task)
  const input = readTaskInput<{ dataset_id?: string | number }>(task)
  const value = output?.dataset_id ?? input?.dataset_id
  return value === undefined || value === null ? null : String(value)
}

function latestEvalOf(dataset: Dataset): Task | undefined {
  return latestEvalTasks.value.get(String(dataset.id))
}

function latestResult(dataset: Dataset): DatasetResult | undefined {
  return dataset.results[0]
}

function summaryOf(dataset: Dataset): MetricEntry[] {
  const summary = latestResult(dataset)?.summary
  if (!summary) return []
  return Object.entries(summary)
    .map(([name, score]) => ({ name, score }))
    .sort((a, b) => a.name.localeCompare(b.name))
}

function evalStatusLabel(dataset: Dataset): string {
  const task = latestEvalOf(dataset)
  if (task && (task.status === 'waiting' || task.status === 'running')) {
    return TASK_STATUS_LABELS[task.status]
  }
  const result = latestResult(dataset)
  if (!result) return ''
  return result.error ? '失败' : '已完成'
}

function evalStatusType(dataset: Dataset): 'primary' | 'success' | 'danger' | 'info' {
  const task = latestEvalOf(dataset)
  if (task && (task.status === 'waiting' || task.status === 'running')) {
    return taskStatusTagType(task.status)
  }
  const result = latestResult(dataset)
  if (!result) return 'info'
  return result.error ? 'danger' : 'success'
}

function evalProgressText(dataset: Dataset): string {
  const task = latestEvalOf(dataset)
  if (task && (task.status === 'waiting' || task.status === 'running')) {
    return latestProgress(task)
  }
  const result = latestResult(dataset)
  if (!result) return ''
  if (result.error) return result.error
  return `报告 #${result.id}`
}

async function loadDatasets() {
  loading.value = true
  try {
    datasets.value = await fetchDatasets({ name: keyword.value.trim() || undefined })
  } catch (error) {
    ElMessage.error(messageOf(error))
  } finally {
    loading.value = false
  }
}

function openCreate() {
  form.value = { documentId: documentOptions.value[0]?.value ?? '', size: 10 }
  createVisible.value = true
}

async function submitCreate() {
  const documentId = form.value.documentId.trim()
  if (!documentId) {
    ElMessage.warning('请选择或填写文档 ID')
    return
  }
  if (!form.value.size || form.value.size < 1) {
    ElMessage.warning('用例条数至少为 1')
    return
  }
  creating.value = true
  try {
    await createDataset({ document_id: documentId, size: form.value.size })
    createVisible.value = false
    ElMessage.success('生成任务已提交，完成后数据集会自动出现在列表中')
    await taskStore.refresh()
  } catch (error) {
    ElMessage.error(messageOf(error))
  } finally {
    creating.value = false
  }
}

async function runEvaluation(dataset: Dataset) {
  try {
    await ElMessageBox.confirm(
      `对数据集「${dataset.name}」发起评测？评测会重新跑一遍检索问答并覆盖旧报告。`,
      '发起评测',
      { type: 'warning', confirmButtonText: '开始评测', cancelButtonText: '取消' },
    )
  } catch {
    return
  }
  evaluatingId.value = dataset.id
  try {
    await evaluateDataset(dataset.id)
    ElMessage.success('评测任务已提交')
    await taskStore.refresh()
  } catch (error) {
    ElMessage.error(messageOf(error))
  } finally {
    evaluatingId.value = null
  }
}

async function removeDataset(dataset: Dataset) {
  try {
    await ElMessageBox.confirm(
      `删除数据集「${dataset.name}」？用例与评测报告会一并删除。`,
      '删除数据集',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return
  }
  try {
    await deleteDataset(dataset.id)
    datasets.value = datasets.value.filter((item) => item.id !== dataset.id)
    ElMessage.success('数据集已删除')
  } catch (error) {
    ElMessage.error(messageOf(error))
  }
}

function openReport(dataset: Dataset) {
  reportDataset.value = dataset
  reportVisible.value = true
}

function barWidth(score: number): string {
  const normalized = Math.max(0, Math.min(1, score))
  return `${(normalized * 100).toFixed(1)}%`
}

function datasetCreatedCount(): number {
  return taskStore.tasks.filter(
    (task) => task.type === 'dataset_create' && task.status === 'success',
  ).length
}

function evalSuccessCount(): number {
  return taskStore.tasks.filter((task) => task.type === 'dataset_eval' && task.status === 'success')
    .length
}

watch(datasetCreatedCount, (next, previous) => {
  if (next > previous && previous > 0) {
    ElMessage.success('数据集生成完成')
    void loadDatasets()
  }
})

watch(evalSuccessCount, (next, previous) => {
  if (next > previous && previous > 0) {
    ElMessage.success('评测完成，报告已更新')
    void loadDatasets()
  }
})

onMounted(async () => {
  await loadDatasets()
  if (!taskStore.tasks.length) await taskStore.refresh()
})
</script>

<template>
  <div class="page">
    <header class="page-header">
      <div>
        <h1 class="page-title">评估数据集</h1>
        <p class="page-sub">按已入库文档生成测试用例，用 Ragas 指标衡量检索问答质量</p>
      </div>
      <div class="page-actions">
        <el-button :loading="loading" @click="loadDatasets">
          <AppIcon name="refresh" :size="16" />
          <span class="btn-label">刷新</span>
        </el-button>
        <el-button type="primary" @click="openCreate">
          <AppIcon name="plus" :size="16" />
          <span class="btn-label">创建数据集</span>
        </el-button>
      </div>
    </header>

    <section class="stat-strip surface-card">
      <div class="stat">
        <span class="stat-value num">{{ datasets.length }}</span>
        <span class="stat-label">数据集</span>
      </div>
      <div class="stat">
        <span class="stat-value num">{{ totalItems }}</span>
        <span class="stat-label">测试用例</span>
      </div>
      <div class="stat">
        <span class="stat-value num">{{ totalResults }}</span>
        <span class="stat-label">评测报告</span>
      </div>
      <div class="stat">
        <span class="stat-value num">{{ activeEvaluationCount }}</span>
        <span class="stat-label">评测任务进行中</span>
      </div>
    </section>

    <section class="table-panel surface-card">
      <div class="table-toolbar">
        <el-input
          v-model="keyword"
          class="search-input"
          placeholder="按名称搜索数据集"
          clearable
          @keydown.enter="loadDatasets"
          @clear="loadDatasets"
        >
          <template #prefix>
            <AppIcon name="search" :size="15" />
          </template>
        </el-input>
        <el-button @click="loadDatasets">搜索</el-button>
      </div>

      <el-table
        v-loading="loading"
        :data="datasets"
        row-key="id"
        empty-text="还没有评估数据集，先基于已入库文档创建一个"
      >
        <el-table-column type="expand">
          <template #default="{ row }">
            <div class="expand-body">
              <template v-if="summaryOf(row).length">
                <p class="expand-title">最近一次评测 · 指标均分</p>
                <div class="metric-grid">
                  <div v-for="metric in summaryOf(row)" :key="metric.name" class="metric">
                    <div class="metric-head">
                      <span class="metric-name mono">{{ metric.name }}</span>
                      <span class="metric-score num">{{ formatScore(metric.score) }}</span>
                    </div>
                    <div class="metric-track">
                      <span class="metric-bar" :style="{ width: barWidth(metric.score) }" />
                    </div>
                  </div>
                </div>
              </template>
              <p v-else class="expand-empty">暂无评测报告。点击「发起评测」后，结果会显示在这里。</p>

              <div v-if="row.items.length" class="case-block">
                <p class="expand-title">测试用例 · {{ row.items.length }} 条</p>
                <p v-for="item in row.items.slice(0, 5)" :key="item.id" class="case-item">
                  {{ item.question }}
                </p>
                <p v-if="row.items.length > 5" class="case-more">
                  还有 {{ row.items.length - 5 }} 条用例…
                </p>
              </div>
            </div>
          </template>
        </el-table-column>

        <el-table-column label="数据集" min-width="240">
          <template #default="{ row }">
            <div class="name-cell">
              <span class="name-text">{{ row.name }}</span>
              <span class="name-id mono">#{{ row.id }}</span>
            </div>
          </template>
        </el-table-column>

        <el-table-column label="用例" width="90" align="right">
          <template #default="{ row }">
            <span class="num">{{ row.items.length }}</span>
          </template>
        </el-table-column>

        <el-table-column label="报告" width="90" align="right">
          <template #default="{ row }">
            <span class="num">{{ row.results.length }}</span>
          </template>
        </el-table-column>

        <el-table-column label="最近评测" min-width="210">
          <template #default="{ row }">
            <div v-if="evalStatusLabel(row)" class="eval-cell">
              <el-tag size="small" :type="evalStatusType(row)" effect="light">
                {{ evalStatusLabel(row) }}
              </el-tag>
              <span class="eval-progress">{{ evalProgressText(row) }}</span>
            </div>
            <span v-else class="muted">—</span>
          </template>
        </el-table-column>

        <el-table-column label="操作" width="230" align="right">
          <template #default="{ row }">
            <el-button
              text
              type="primary"
              :loading="evaluatingId === row.id"
              @click="runEvaluation(row)"
            >
              发起评测
            </el-button>
            <el-button text :disabled="!latestResult(row)" @click="openReport(row)">
              报告
            </el-button>
            <el-button text type="danger" @click="removeDataset(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </section>

    <el-dialog
      v-model="createVisible"
      title="创建评估数据集"
      width="560px"
      :close-on-click-modal="false"
    >
      <div class="dialog-body">
        <div class="field">
          <label class="field-label">来源文档</label>
          <el-select
            v-model="form.documentId"
            class="w-full"
            filterable
            allow-create
            default-first-option
            placeholder="选择已入库文档，或直接填写文档 ID"
          >
            <el-option
              v-for="option in documentOptions"
              :key="option.value"
              :label="option.label"
              :value="option.value"
            />
          </el-select>
          <p class="field-hint">
            后台按 document_id 读取该文档解析出的 Markdown 并切分生成用例；文档需已完成入库。
          </p>
          <p v-if="!documentOptions.length" class="field-hint is-warning">
            暂无已入库文档可选，可手动粘贴文档 ID。
          </p>
        </div>

        <div class="field field-inline">
          <label class="field-label">用例条数</label>
          <el-input-number v-model="form.size" :min="1" :max="100" :step="1" />
          <span class="field-hint">默认 10 条，生成在后台执行，可在任务队列查看进度。</span>
        </div>
      </div>

      <template #footer>
        <el-button @click="createVisible = false">取消</el-button>
        <el-button type="primary" :loading="creating" @click="submitCreate">
          提交生成
        </el-button>
      </template>
    </el-dialog>

    <el-dialog
      v-model="reportVisible"
      :title="reportDataset?.name || '评测报告'"
      width="720px"
      top="8vh"
    >
      <div v-if="reportResult" class="report-body">
        <div class="report-meta">
          <span class="mono">报告 #{{ reportResult.id }}</span>
          <el-tag size="small" :type="reportResult.error ? 'danger' : 'success'" effect="light">
            {{ reportResult.error ? '评测失败' : '已完成' }}
          </el-tag>
        </div>

        <p v-if="reportResult.error" class="report-error">{{ reportResult.error }}</p>

        <div v-if="reportMetrics.length" class="metric-grid report-metrics">
          <div v-for="metric in reportMetrics" :key="metric.name" class="metric">
            <div class="metric-head">
              <span class="metric-name mono">{{ metric.name }}</span>
              <span class="metric-score num">{{ formatScore(metric.score) }}</span>
            </div>
            <div class="metric-track">
              <span class="metric-bar" :style="{ width: barWidth(metric.score) }" />
            </div>
          </div>
        </div>

        <div class="report-content">
          <p class="expand-title">评测解读</p>
          <pre class="report-markdown">{{ reportResult.content || '暂无报告正文' }}</pre>
        </div>
      </div>

      <div v-else class="summary-empty">
        <p>该数据集还没有评测报告。</p>
        <p v-if="reportTask" class="muted">{{ latestProgress(reportTask) }}</p>
      </div>

      <template #footer>
        <el-button @click="reportVisible = false">关闭</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.page-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 20px;
}

.page-actions {
  display: flex;
  gap: 8px;
  flex: none;
}

.btn-label {
  margin-left: 6px;
}

.stat-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin-bottom: 16px;
  overflow: hidden;
}

.stat {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 18px 24px;
}

.stat + .stat {
  border-left: 1px solid var(--hairline);
}

.stat-value {
  font-size: 28px;
  font-weight: 600;
  line-height: 1.2;
  color: var(--ink);
}

.stat-label {
  font-size: 12px;
  color: var(--muted);
}

.table-panel {
  padding: 16px 16px 4px;
}

.table-toolbar {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}

.search-input {
  width: 280px;
}

.name-cell {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.name-text {
  font-size: 13px;
  font-weight: 600;
  color: var(--ink);
}

.name-id {
  font-size: 11px;
  color: var(--muted);
}

.eval-cell {
  display: flex;
  flex-direction: column;
  gap: 4px;
  align-items: flex-start;
}

.eval-progress {
  font-size: 11px;
  color: var(--muted);
  word-break: break-all;
}

.expand-body {
  padding: 8px 32px 16px;
}

.expand-title {
  margin-bottom: 10px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  color: var(--muted);
}

.expand-empty {
  font-size: 12px;
  color: var(--muted);
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 12px 24px;
}

.metric-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}

.metric-name {
  font-size: 12px;
  color: var(--text);
}

.metric-score {
  font-size: 12px;
  font-weight: 600;
  color: var(--ink);
}

.metric-track {
  height: 6px;
  margin-top: 6px;
  border-radius: 3px;
  background: var(--surface);
  overflow: hidden;
}

.metric-bar {
  display: block;
  height: 100%;
  border-radius: 3px;
  background: var(--accent);
  transition: width 320ms ease;
}

.case-block {
  margin-top: 18px;
  padding-top: 14px;
  border-top: 1px solid var(--hairline);
}

.case-item {
  font-size: 12px;
  line-height: 1.7;
  color: var(--text);
}

.case-item + .case-item {
  margin-top: 4px;
}

.case-more {
  margin-top: 4px;
  font-size: 11px;
  color: var(--muted);
}

.dialog-body {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.field-inline {
  flex-direction: row;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.field-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--ink);
}

.field-hint {
  font-size: 11px;
  color: var(--muted);
}

.field-hint.is-warning {
  color: var(--warning);
}

.w-full {
  width: 100%;
}

.report-body {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.report-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  font-size: 12px;
  color: var(--muted);
}

.report-error {
  padding: 10px 12px;
  border-radius: 8px;
  background: var(--tint-peach);
  color: var(--danger);
  font-size: 12px;
}

.report-metrics {
  padding-bottom: 4px;
}

.report-content {
  padding-top: 12px;
  border-top: 1px solid var(--hairline);
}

.report-markdown {
  max-height: 42vh;
  overflow: auto;
  margin: 0;
  padding: 12px 14px;
  border: 1px solid var(--hairline);
  border-radius: 8px;
  background: var(--surface-soft);
  font-family: Aptos, 'Microsoft YaHei', 'PingFang SC', sans-serif;
  font-size: 12px;
  line-height: 1.75;
  color: var(--text);
  white-space: pre-wrap;
  word-break: break-word;
}

.summary-empty {
  padding: 16px 0;
  font-size: 13px;
  color: var(--text);
}

.summary-empty .muted {
  margin-top: 6px;
  font-size: 12px;
}

@media (max-width: 900px) {
  .page-header {
    flex-direction: column;
  }

  .stat-strip {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .stat:nth-child(3) {
    border-left: 0;
    border-top: 1px solid var(--hairline);
  }

  .stat:nth-child(4) {
    border-top: 1px solid var(--hairline);
  }

  .table-toolbar {
    flex-wrap: wrap;
  }

  .search-input {
    width: 100%;
  }
}
</style>
