<script setup lang="ts">
import { ref, watch } from 'vue'
import AppIcon from '@/components/AppIcon.vue'

const props = defineProps<{ title?: string; sources: string[] }>()

const open = defineModel<boolean>({ default: false })

const resolvedUrl = ref('')
const loading = ref(false)
const error = ref('')
const frameKey = ref(0)

type ProbeResult = 'ok' | 'fail' | 'unknown'

async function probe(url: string): Promise<ProbeResult> {
  if (!url.startsWith('/')) return 'unknown'
  try {
    const response = await fetch(url, { method: 'HEAD', cache: 'no-store' })
    if (!response.ok) return 'fail'
    const contentType = response.headers.get('content-type') ?? ''
    return contentType.includes('pdf') ? 'ok' : 'fail'
  } catch {
    return 'fail'
  }
}

async function resolveSource() {
  resolvedUrl.value = ''
  error.value = ''
  if (!props.sources.length) {
    error.value = '没有可用的 PDF 地址：后端未返回文件链接，本地 storage 目录也没有对应文件'
    return
  }

  loading.value = true
  try {
    for (const source of props.sources) {
      const result = await probe(source)
      if (result === 'fail') continue
      resolvedUrl.value = source
      frameKey.value += 1
      return
    }
    error.value = 'PDF 无法访问：对象存储地址不可用，本地开发通道也没有找到该文件'
  } finally {
    loading.value = false
  }
}

function reload() {
  if (resolvedUrl.value) frameKey.value += 1
  else void resolveSource()
}

function openExternal() {
  if (!resolvedUrl.value) return
  window.open(resolvedUrl.value, '_blank', 'noopener')
}

watch(
  [open, () => props.sources],
  () => {
    if (open.value) void resolveSource()
  },
  { deep: true },
)
</script>

<template>
  <el-dialog
    v-model="open"
    :title="title || 'PDF 预览'"
    width="72%"
    top="5vh"
    destroy-on-close
    class="pdf-dialog"
  >
    <div v-loading="loading" class="pdf-body">
      <iframe
        v-if="resolvedUrl"
        :key="frameKey"
        :src="resolvedUrl"
        class="pdf-frame"
        title="PDF 预览"
      />
      <div v-else-if="error" class="pdf-error">
        <span class="error-mark"><AppIcon name="file" :size="22" /></span>
        <p class="error-title">{{ error }}</p>
        <p class="error-hint">
          开发环境会自动读取后端 storage 目录；生产环境需要后端提供可访问的 PDF 下载地址。
        </p>
        <div class="error-links">
          <a
            v-for="source in sources"
            :key="source"
            :href="source"
            target="_blank"
            rel="noopener noreferrer"
          >
            {{ source }}
          </a>
          <span v-if="!sources.length" class="muted">无候选地址</span>
        </div>
      </div>
    </div>

    <template #footer>
      <div class="dialog-foot">
        <span class="foot-hint">
          {{ resolvedUrl ? '可用浏览器 PDF 工具栏缩放和翻页' : '等待可用地址' }}
        </span>
        <div class="foot-actions">
          <el-button text :disabled="!resolvedUrl" @click="openExternal">新窗口打开</el-button>
          <el-button :disabled="loading" @click="reload">重新加载</el-button>
          <el-button @click="open = false">关闭</el-button>
        </div>
      </div>
    </template>
  </el-dialog>
</template>

<style scoped>
.pdf-body {
  height: 74vh;
  min-height: 360px;
  border: 1px solid var(--hairline);
  border-radius: 8px;
  background: var(--surface);
  overflow: hidden;
}

.pdf-frame {
  width: 100%;
  height: 100%;
  border: 0;
  display: block;
}

.pdf-error {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding: 24px;
  text-align: center;
}

.error-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 44px;
  height: 44px;
  border-radius: 10px;
  background: var(--tint-peach);
  color: var(--warning);
}

.error-title {
  margin-top: 12px;
  font-size: 14px;
  font-weight: 600;
  color: var(--ink);
}

.error-hint {
  margin-top: 6px;
  font-size: 12px;
  color: var(--muted);
  max-width: 460px;
}

.error-links {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-top: 14px;
  font-size: 11px;
  word-break: break-all;
}

.dialog-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.foot-hint {
  font-size: 11px;
  color: var(--muted);
}

.foot-actions {
  display: flex;
  align-items: center;
}
</style>
