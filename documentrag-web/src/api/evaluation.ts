import { del, get, post } from '@/plugins/request'
import type { Dataset, Task } from './types'

export interface CreateDatasetPayload {
  document_id: string
  size: number
}

export function createDataset(payload: CreateDatasetPayload): Promise<Task> {
  return post<Task>('/evaluation/dataset', payload)
}

export function fetchDatasets(query?: { name?: string; limit?: number }): Promise<Dataset[]> {
  return get<Dataset[]>('/evaluation/dataset', { name: query?.name || undefined, limit: query?.limit })
}

export function deleteDataset(datasetId: string): Promise<void> {
  return del<void>(`/evaluation/dataset/${encodeURIComponent(datasetId)}`)
}

export function evaluateDataset(datasetId: string): Promise<Task> {
  return post<Task>(`/evaluation/${encodeURIComponent(datasetId)}`)
}
