import type { SnowflakeId } from '@/plugins/request'

export type TaskType = 'document_ingest' | 'dataset_create' | 'dataset_eval'
export type TaskStatus = 'waiting' | 'running' | 'success' | 'error'
export type StepStatus = 'running' | 'success' | 'error'
export type ProgressStatus = StepStatus | 'waiting'

export interface TaskProgressStep {
  message: string
  status: ProgressStatus
  error: string | null
}

export interface Task {
  id: SnowflakeId
  type: TaskType
  status: TaskStatus
  created_at: string
  error: string | null
  inputs: Record<string, unknown>
  outputs: Record<string, unknown>
}

export interface Chunk {
  title: string
  content: string
  source: 'local' | 'online' | string
  id: SnowflakeId | number | null
  url: string | null
  filename: string | null
}

export interface Step {
  message: string
  status: StepStatus
  error: string | null
  answer: string | null
  chunks: Chunk[] | null
}

export interface Turn {
  id: SnowflakeId
  created_at: string
  steps: Step[]
}

export interface SessionSummary {
  id: SnowflakeId
  thread_id: SnowflakeId
  title: string
  created_at: string
  turns: number
}

export interface SessionDetail {
  id: SnowflakeId
  thread_id: SnowflakeId
  title: string
  created_at: string
  turns: Turn[]
}

export interface DatasetItem {
  id: SnowflakeId
  question: string
  reference: string
  reference_contexts: string[]
}

export interface DatasetResult {
  id: SnowflakeId
  summary: Record<string, number>
  content: string
  error: string | null
}

export interface Dataset {
  id: SnowflakeId
  name: string
  thread_id: SnowflakeId
  items: DatasetItem[]
  results: DatasetResult[]
}

export interface ObjectInfo {
  object_name: string
  document_id: SnowflakeId | null
  size: number
  content_type: string
  last_modified: string | null
  etag: string
  url: string
}

export interface IngestInput {
  document_id?: SnowflakeId
  filename?: string
  pdf_path?: string
  object_name?: string
}

export interface IngestOutput {
  document_id?: SnowflakeId
  file_url?: string
  file_path?: string
  content?: string
}

export interface DatasetCreateOutput {
  dataset_id?: SnowflakeId
  thread_id?: SnowflakeId
  name?: string
  items?: number
}

export interface DatasetEvalOutput {
  dataset_id?: SnowflakeId
  thread_id?: SnowflakeId
  result_id?: SnowflakeId | null
  summary?: Record<string, number>
}
