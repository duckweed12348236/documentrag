import { del, get, post } from '@/plugins/request'
import type { Task, TaskStatus, TaskType } from './types'

export interface TaskQuery {
  status?: TaskStatus
  type?: TaskType
  limit?: number
}

export function fetchTasks(query?: TaskQuery): Promise<Task[]> {
  return get<Task[]>('/task', {
    status: query?.status,
    type: query?.type,
    limit: query?.limit,
  })
}

export function retryTask(taskId: string): Promise<Task> {
  return post<Task>(`/task/${encodeURIComponent(taskId)}/retry`)
}

export function deleteTask(taskId: string): Promise<void> {
  return del<void>(`/task/${encodeURIComponent(taskId)}`)
}
