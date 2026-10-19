<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import AppIcon from '@/components/AppIcon.vue'
import { useTaskStore } from '@/stores/task'
import type { Task } from '@/api/types'
import { formatDateTime, messageOf } from '@/utils/format'
import {
  TASK_STATUS_LABELS,
  TASK_TYPE_LABELS,
  latestProgress,
  progressOf,
  taskStatusTagType,
} from '@/utils/task'

const open = defineModel<boolean>({ default: false })

const taskStore = useTaskStore()
const retryingId = ref<string | null>(null)
const deletingId = ref<string | null>(null)

async function handleRetry(task: Task) {
  retryingId.value = task.id
  try {
    await taskStore.retry(task.id)
    ElMessage.success('任务已重新排队')
  } catch (error) {
    ElMessage.error(messageOf(error))
  } finally {
    retryingId.value = null
  }
}

async function handleDelete(task: Task) {
  try {
    await ElMessageBox.confirm(
      `删除任务记录「${TASK_TYPE_LABELS[task.type] ?? task.type} #${task.id}」？仅移除状态快照，不会取消后台任务。`,
      '删除任务',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return
  }
  deletingId.value = task.id
  try {
    await taskStore.remove(task.id)
    ElMessage.success('任务记录已删除')
  } catch (error) {
    ElMessage.error(messageOf(error))
  } finally {
    deletingId.value = null
  }
}
</script>

<template>
  <el-drawer v-model="open" size="440px" :with-header="false" class="task-drawer">
    <div class="drawer-head">
      <div>
        <h2 class="drawer-title">任务队列</h2>
        <p class="drawer-sub">
          {{ taskStore.activeCount }} 个任务进行中 · 每 3 秒自动刷新
        </p>
      </div>
      <div class="drawer-actions">
        <el-button text :loading="taskStore.loading" @click="taskStore.refresh()">
          <AppIcon name="refresh" :size="16" />
        </el-button>
        <el-button text @click="open = false">
          <AppIcon name="close" :size="16" />
        </el-button>
      </div>
    </div>

    <div v-if="taskStore.error" class="drawer-error">{{ taskStore.error }}</div>

    <div class="task-list">
      <el-empty
        v-if="!taskStore.tasks.length && !taskStore.loading"
        description="还没有任务记录"
        :image-size="72"
      />
      <article v-for="task in taskStore.tasks" :key="task.id" class="task-card">
        <div class="task-row">
          <span class="task-type">{{ TASK_TYPE_LABELS[task.type] ?? task.type }}</span>
          <el-tag size="small" :type="taskStatusTagType(task.status)" effect="light">
            {{ TASK_STATUS_LABELS[task.status] ?? task.status }}
          </el-tag>
          <span class="task-time num">{{ formatDateTime(task.created_at) }}</span>
        </div>

        <p class="task-progress">
          {{ latestProgress(task) }}
          <span v-if="progressOf(task).length" class="task-step-count">
            · {{ progressOf(task).length }} 步
          </span>
        </p>

        <p v-if="task.error" class="task-error">{{ task.error }}</p>

        <div class="task-foot">
          <span class="mono task-id">#{{ task.id }}</span>
          <div class="task-actions">
            <el-button
              v-if="task.status === 'error'"
              size="small"
              text
              type="primary"
              :loading="retryingId === task.id"
              @click="handleRetry(task)"
            >
              重试
            </el-button>
            <el-button
              v-if="task.status !== 'running'"
              size="small"
              text
              type="danger"
              :loading="deletingId === task.id"
              @click="handleDelete(task)"
            >
              删除
            </el-button>
          </div>
        </div>
      </article>
    </div>
  </el-drawer>
</template>

<style scoped>
.drawer-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--hairline);
}

.drawer-title {
  font-family: 'Aptos Display', Aptos, 'Microsoft YaHei', sans-serif;
  font-size: 20px;
  font-weight: 600;
  color: var(--ink);
}

.drawer-sub {
  margin-top: 4px;
  font-size: 12px;
  color: var(--muted);
}

.drawer-actions {
  display: flex;
  gap: 2px;
}

.drawer-error {
  margin-top: 12px;
  padding: 8px 12px;
  border-radius: 8px;
  background: var(--tint-peach);
  color: var(--warning);
  font-size: 12px;
}

.task-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 16px;
}

.task-card {
  border: 1px solid var(--hairline);
  border-radius: 10px;
  padding: 12px 14px;
  background: var(--canvas);
  transition: background 160ms ease;
}

.task-card:hover {
  background: var(--surface-soft);
}

.task-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.task-type {
  font-size: 13px;
  font-weight: 600;
  color: var(--ink);
}

.task-time {
  margin-left: auto;
  font-size: 11px;
  color: var(--muted);
}

.task-progress {
  margin-top: 8px;
  font-size: 12px;
  color: var(--text);
}

.task-step-count {
  color: var(--muted);
}

.task-error {
  margin-top: 6px;
  font-size: 12px;
  color: var(--danger);
  word-break: break-all;
}

.task-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 8px;
}

.task-actions {
  display: flex;
  align-items: center;
  gap: 2px;
}

.task-id {
  font-size: 11px;
  color: var(--muted);
}
</style>
