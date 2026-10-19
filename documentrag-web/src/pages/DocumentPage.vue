<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import AppIcon from '@/components/AppIcon.vue'
import PdfPreview from '@/components/PdfPreview.vue'
import { deleteDocument, fetchDocuments, uploadDocument } from '@/api/document'
import type { ObjectInfo } from '@/api/types'
import { formatBytes, formatDateTime, messageOf } from '@/utils/format'
import { useTaskStore } from '@/stores/task'

const MAX_SIZE = 100 * 1024 * 1024

const taskStore = useTaskStore()

const storageObjects = ref<ObjectInfo[]>([])
const loading = ref(false)
const keyword = ref('')
const storageError = ref<string | null>(null)

const uploading = ref(false)
const uploadPercent = ref(0)
const uploadingName = ref('')
const dragActive = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)
const deletingName = ref<string | null>(null)

const previewVisible = ref(false)
const previewTitle = ref('')
const previewSources = ref<string[]>([])

const documents = computed(() => storageObjects.value)

const visibleDocuments = computed(() => {
  const query = keyword.value.trim().toLowerCase()
  if (!query) return documents.value
  return documents.value.filter((item) =>
    [baseName(item.object_name), item.object_name, item.document_id]
      .filter((value): value is string => Boolean(value))
      .some((value) => value.toLowerCase().includes(query)),
  )
})

const totalSize = computed(() => documents.value.reduce((sum, item) => sum + item.size, 0))

const parsingCount = computed(
  () =>
    taskStore.tasks.filter(
      (task) =>
        task.type === 'document_ingest' && (task.status === 'waiting' || task.status === 'running'),
    ).length,
)

async function loadDocuments() {
  loading.value = true
  storageError.value = null
  try {
    storageObjects.value = await fetchDocuments()
  } catch (error) {
    storageError.value = messageOf(error)
    storageObjects.value = []
  } finally {
    loading.value = false
  }
}

function triggerPick() {
  if (uploading.value) return
  fileInput.value?.click()
}

function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (file) void handleFile(file)
}

function onDrop(event: DragEvent) {
  dragActive.value = false
  const file = event.dataTransfer?.files?.[0]
  if (file) void handleFile(file)
}

function onDropzoneKeydown(event: KeyboardEvent) {
  if (event.key !== 'Enter' && event.key !== ' ') return
  event.preventDefault()
  triggerPick()
}

async function handleFile(file: File) {
  if (uploading.value) return
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    ElMessage.warning('仅支持上传 PDF 文档')
    return
  }
  if (file.size > MAX_SIZE) {
    ElMessage.warning('PDF 大小不能超过 100 MB')
    return
  }

  uploading.value = true
  uploadPercent.value = 0
  uploadingName.value = file.name
  try {
    await uploadDocument(file, (percent) => {
      uploadPercent.value = percent
    })
    ElMessage.success('PDF 已上传，入库任务已提交，解析完成后会出现在列表中')
    await Promise.all([loadDocuments(), taskStore.refresh()])
  } catch (error) {
    ElMessage.error(messageOf(error))
  } finally {
    uploading.value = false
    uploadPercent.value = 0
    uploadingName.value = ''
  }
}

async function removeDocument(item: ObjectInfo) {
  try {
    await ElMessageBox.confirm(
      `删除对象「${item.object_name}」？该操作只移除对象存储文件，已入库的向量数据不会一并清理。`,
      '删除文件',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return
  }
  deletingName.value = item.object_name
  try {
    await deleteDocument(item.object_name)
    storageObjects.value = storageObjects.value.filter(
      (doc) => doc.object_name !== item.object_name,
    )
    ElMessage.success('文件已删除')
  } catch (error) {
    ElMessage.error(messageOf(error))
  } finally {
    deletingName.value = null
  }
}

function baseName(filePath: string): string {
  const segments = filePath.split(/[\\/]/)
  return segments[segments.length - 1] || filePath
}

function openPreview(item: ObjectInfo) {
  const sources: string[] = []
  const push = (url: string | null | undefined) => {
    if (url && !sources.includes(url)) sources.push(url)
  }

  push(item.url)
  push(`/local-storage/${encodeURIComponent(baseName(item.object_name))}`)

  previewTitle.value = baseName(item.object_name)
  previewSources.value = sources
  previewVisible.value = true
}

onMounted(async () => {
  await loadDocuments()
})
</script>

<template>
  <div class="page">
    <header class="page-header">
      <div>
        <h1 class="page-title">文档管理</h1>
        <p class="page-sub">上传 PDF 后由后台完成解析、切分与向量入库，列表展示对象存储中的可预览文档</p>
      </div>
      <div class="page-actions">
        <el-button :loading="loading" @click="loadDocuments">
          <AppIcon name="refresh" :size="16" />
          <span class="btn-label">刷新</span>
        </el-button>
      </div>
    </header>

    <section class="upload-panel surface-card">
      <div
        class="dropzone"
        :class="{ 'is-dragging': dragActive, 'is-uploading': uploading }"
        role="button"
        tabindex="0"
        @click="triggerPick"
        @keydown="onDropzoneKeydown"
        @dragover.prevent="dragActive = true"
        @dragenter.prevent="dragActive = true"
        @dragleave.prevent="dragActive = false"
        @drop.prevent="onDrop"
      >
        <span class="drop-mark"><AppIcon name="upload" :size="22" /></span>
        <div class="drop-copy">
          <p class="drop-title">拖拽 PDF 到此处上传</p>
          <p class="drop-hint">仅支持 PDF（扩展名 .pdf 且内容以 %PDF- 开头），单个文件不超过 100 MB</p>
        </div>
        <button type="button" class="drop-trigger" @click.stop="triggerPick">
          选择文件
        </button>
        <input
          ref="fileInput"
          type="file"
          accept="application/pdf,.pdf"
          hidden
          @change="onFileChange"
        />
      </div>

      <div v-if="uploading" class="upload-progress">
        <div class="progress-head">
          <span class="progress-name">{{ uploadingName }}</span>
          <span class="progress-value num">{{ uploadPercent }}%</span>
        </div>
        <el-progress :percentage="uploadPercent" :stroke-width="6" :show-text="false" />
        <p class="progress-hint">上传完成后会自动创建入库任务，解析进度见任务队列。</p>
      </div>
    </section>

    <section class="stat-strip surface-card">
      <div class="stat">
        <span class="stat-value num">{{ documents.length }}</span>
        <span class="stat-label">已解析文档</span>
      </div>
      <div class="stat">
        <span class="stat-value num">{{ formatBytes(totalSize) }}</span>
        <span class="stat-label">对象存储占用</span>
      </div>
      <div class="stat">
        <span class="stat-value num">{{ parsingCount }}</span>
        <span class="stat-label">入库任务进行中</span>
      </div>
    </section>

    <section class="table-panel surface-card">
      <div class="table-toolbar">
        <el-input
          v-model="keyword"
          class="search-input"
          placeholder="按文件名、对象键或文档 ID 筛选"
          clearable
        >
          <template #prefix>
            <AppIcon name="search" :size="15" />
          </template>
        </el-input>
        <span class="toolbar-hint">对象键形如 &lt;文档 ID&gt;/&lt;原文件名&gt;</span>
      </div>

      <div v-if="storageError" class="storage-banner is-error">
        对象存储文件列表加载失败：{{ storageError }}
      </div>

      <el-table
        v-loading="loading"
        :data="visibleDocuments"
        row-key="object_name"
        empty-text="对象存储中还没有可展示的文档"
      >
        <el-table-column label="文件" min-width="320">
          <template #default="{ row }">
            <div class="file-cell">
              <span class="file-name">{{ baseName(row.object_name) }}</span>
              <span class="file-key mono">{{ row.object_name }}</span>
              <span v-if="row.document_id" class="file-key mono">文档 ID · {{ row.document_id }}</span>
            </div>
          </template>
        </el-table-column>

        <el-table-column label="大小" width="100" align="right">
          <template #default="{ row }">
            <span class="num">{{ formatBytes(row.size) }}</span>
          </template>
        </el-table-column>

        <el-table-column label="最后修改" width="170">
          <template #default="{ row }">
            <span class="num muted">{{ formatDateTime(row.last_modified) }}</span>
          </template>
        </el-table-column>

        <el-table-column label="操作" width="150" align="right">
          <template #default="{ row }">
            <el-button text type="primary" @click="openPreview(row)">预览</el-button>
            <el-button
              text
              type="danger"
              :loading="deletingName === row.object_name"
              @click="removeDocument(row)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </section>

    <PdfPreview v-model="previewVisible" :title="previewTitle" :sources="previewSources" />
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

.upload-panel {
  padding: 16px;
  margin-bottom: 16px;
}

.dropzone {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 18px 20px;
  border: 1px dashed #d6d2cc;
  border-radius: 10px;
  background: var(--surface-soft);
  cursor: pointer;
  transition:
    border-color 160ms ease,
    background 160ms ease;
}

.dropzone:hover,
.dropzone.is-dragging {
  border-color: var(--accent);
  background: var(--tint-lavender);
}

.dropzone.is-uploading {
  cursor: progress;
  opacity: 0.7;
}

.dropzone:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

.drop-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 44px;
  height: 44px;
  flex: none;
  border-radius: 10px;
  background: var(--canvas);
  border: 1px solid var(--hairline);
  color: var(--accent);
}

.drop-copy {
  flex: 1;
  min-width: 0;
}

.drop-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--ink);
}

.drop-hint {
  margin-top: 2px;
  font-size: 12px;
  color: var(--muted);
}

.drop-trigger {
  flex: none;
  height: 36px;
  padding: 0 16px;
  border: 1px solid var(--hairline);
  border-radius: 8px;
  background: var(--canvas);
  color: var(--text);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition:
    border-color 160ms ease,
    color 160ms ease;
}

.drop-trigger:hover {
  border-color: var(--accent);
  color: var(--accent);
}

.upload-progress {
  margin-top: 14px;
  padding: 12px 14px;
  border-radius: 8px;
  background: var(--surface-soft);
}

.progress-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
  font-size: 12px;
}

.progress-name {
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.progress-value {
  color: var(--accent);
  font-weight: 600;
}

.progress-hint {
  margin-top: 8px;
  font-size: 11px;
  color: var(--muted);
}

.stat-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin-bottom: 16px;
}

.stat {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 16px 24px;
}

.stat + .stat {
  border-left: 1px solid var(--hairline);
}

.stat-value {
  font-size: 24px;
  font-weight: 600;
  line-height: 1.25;
  color: var(--ink);
  font-variant-numeric: tabular-nums;
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
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}

.search-input {
  width: 300px;
}

.toolbar-hint {
  margin-left: auto;
  font-size: 11px;
  color: var(--muted);
}

.storage-banner {
  margin-bottom: 12px;
  padding: 8px 12px;
  border-radius: 8px;
  background: var(--surface-soft);
  color: var(--muted);
  font-size: 12px;
}

.storage-banner.is-error {
  background: var(--tint-peach);
  color: var(--warning);
}

.file-cell {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.file-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--ink);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-key {
  font-size: 11px;
  color: var(--muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

@media (max-width: 900px) {
  .page-header {
    flex-direction: column;
  }

  .dropzone {
    flex-wrap: wrap;
  }

  .drop-copy {
    flex: 1 1 200px;
  }

  .stat-strip {
    grid-template-columns: minmax(0, 1fr);
  }

  .stat + .stat {
    border-left: 0;
    border-top: 1px solid var(--hairline);
  }

  .table-toolbar {
    flex-wrap: wrap;
  }

  .search-input {
    width: 100%;
  }

  .toolbar-hint {
    margin-left: 0;
  }
}
</style>
