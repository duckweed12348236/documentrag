import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { deleteTask, fetchTasks, retryTask } from '@/api/task'
import type { Task } from '@/api/types'
import { messageOf } from '@/utils/format'

const POLL_INTERVAL = 3000

export const useTaskStore = defineStore('task', () => {
  const tasks = ref<Task[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  let timer: ReturnType<typeof setInterval> | null = null
  let pending = false

  const activeTasks = computed(() =>
    tasks.value.filter((task) => task.status === 'waiting' || task.status === 'running'),
  )

  const activeCount = computed(() => activeTasks.value.length)

  async function refresh() {
    if (pending) return
    pending = true
    loading.value = tasks.value.length === 0
    try {
      tasks.value = await fetchTasks()
      error.value = null
    } catch (err) {
      error.value = messageOf(err)
    } finally {
      pending = false
      loading.value = false
    }
  }

  function start() {
    if (timer) return
    void refresh()
    timer = setInterval(() => {
      void refresh()
    }, POLL_INTERVAL)
  }

  function stop() {
    if (!timer) return
    clearInterval(timer)
    timer = null
  }

  async function retry(taskId: string) {
    await retryTask(taskId)
    await refresh()
  }

  async function remove(taskId: string) {
    await deleteTask(taskId)
    await refresh()
  }

  return { tasks, loading, error, activeTasks, activeCount, refresh, start, stop, retry, remove }
})
