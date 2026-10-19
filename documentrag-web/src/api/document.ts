import { del, encodePath, get, upload } from '@/plugins/request'
import type { ObjectInfo, Task } from './types'

export function uploadDocument(
  file: File,
  onProgress?: (percent: number) => void,
): Promise<Task> {
  return upload<Task>('/ingestion', file, {
    timeout: 0,
    onUploadProgress: (event) => {
      if (!event.total) return
      onProgress?.(Math.round((event.loaded / event.total) * 100))
    },
  })
}

export function fetchDocuments(prefix?: string): Promise<ObjectInfo[]> {
  return get<ObjectInfo[]>('/document', { prefix: prefix || undefined })
}

export function deleteDocument(objectName: string): Promise<void> {
  return del<void>(`/document/${encodePath(objectName)}`)
}
