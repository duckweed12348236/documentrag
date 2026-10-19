import type {
  IngestInput,
  IngestOutput,
  Task,
  TaskProgressStep,
  TaskStatus,
  TaskType,
} from '@/api/types'

export const TASK_TYPE_LABELS: Record<TaskType, string> = {
  document_ingest: '文档入库',
  dataset_create: '数据集生成',
  dataset_eval: '数据集评测',
}

export const TASK_STATUS_LABELS: Record<TaskStatus, string> = {
  waiting: '排队中',
  running: '执行中',
  success: '已完成',
  error: '失败',
}

export function taskStatusTagType(status: TaskStatus): 'info' | 'primary' | 'success' | 'danger' {
  if (status === 'running') return 'primary'
  if (status === 'success') return 'success'
  if (status === 'error') return 'danger'
  return 'info'
}

export function latestProgress(task: Task): string {
  const progress = progressOf(task)
  if (progress.length === 0) return '等待执行'
  const last = progress[progress.length - 1]
  return last?.message ?? '等待执行'
}

export function progressOf(task: Task): TaskProgressStep[] {
  const progress = task.outputs?.progress
  if (!Array.isArray(progress)) return []
  return progress as TaskProgressStep[]
}

export function readTaskOutput<T>(task: Task): T | null {
  const output = task.outputs
  if (!output || typeof output !== 'object') return null
  return output as T
}

export function readTaskInput<T>(task: Task): T | null {
  const input = task.inputs
  if (!input || typeof input !== 'object') return null
  return input as T
}

/**
 * 取入库任务的文档 ID：优先用 input / output 的 document_id，
 * 老版本任务没有该字段时回退到 Markdown 文件名（<document_id>.md）。
 */
export function documentIdOf(task: Task): string | null {
  if (task.type !== 'document_ingest') return null
  const input = readTaskInput<IngestInput>(task)
  const output = readTaskOutput<IngestOutput>(task)
  const direct = output?.document_id ?? input?.document_id
  if (direct) return String(direct)
  const filePath = output?.file_path
  if (!filePath) return null
  const segments = filePath.split(/[\\/]/)
  const name = segments[segments.length - 1] ?? ''
  const id = name.replace(/\.md$/i, '')
  return /^[0-9a-f]{32}$/i.test(id) ? id : null
}
