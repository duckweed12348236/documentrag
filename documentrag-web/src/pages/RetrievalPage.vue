<script setup lang="ts">
import { onMounted, nextTick, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import AppIcon from '@/components/AppIcon.vue'
import { askStream, deleteSession, fetchSession, fetchSessions } from '@/api/retrieval'
import { RequestError } from '@/plugins/request'
import type { Chunk, SessionSummary, Step } from '@/api/types'
import { formatDateTime, messageOf } from '@/utils/format'

interface ChatTurn {
  key: string
  question: string | null
  steps: Step[]
  createdAt: string
  pending: boolean
  error: string | null
}

const sessions = ref<SessionSummary[]>([])
const sessionsLoading = ref(false)
const sessionsError = ref<string | null>(null)

const sessionId = ref<string | null>(null)
const sessionThreadId = ref<string | null>(null)
const sessionTitle = ref('新会话')

const turns = ref<ChatTurn[]>([])
const detailLoading = ref(false)
const sending = ref(false)
const draft = ref('')
const abortController = ref<AbortController | null>(null)

const expandedTurns = ref(new Set<string>())
const openChunks = ref(new Set<string>())
const bodyRef = ref<HTMLElement | null>(null)

async function loadSessions() {
  sessionsLoading.value = true
  sessionsError.value = null
  try {
    sessions.value = await fetchSessions(50)
  } catch (error) {
    sessionsError.value = messageOf(error)
  } finally {
    sessionsLoading.value = false
  }
}

function startNewSession() {
  if (sending.value) return
  sessionId.value = null
  sessionThreadId.value = null
  sessionTitle.value = '新会话'
  turns.value = []
  expandedTurns.value = new Set()
  openChunks.value = new Set()
}

async function selectSession(session: SessionSummary) {
  if (sending.value || detailLoading.value) return
  detailLoading.value = true
  try {
    const detail = await fetchSession(session.id)
    sessionId.value = detail.id
    sessionThreadId.value = detail.thread_id
    sessionTitle.value = detail.title
    turns.value = detail.turns.map((turn) => ({
      key: turn.id,
      question: null,
      steps: turn.steps ?? [],
      createdAt: turn.created_at,
      pending: false,
      error: null,
    }))
    expandedTurns.value = new Set()
    openChunks.value = new Set()
    await scrollToBottom()
  } catch (error) {
    ElMessage.error(messageOf(error))
  } finally {
    detailLoading.value = false
  }
}

async function removeSession(session: SessionSummary) {
  try {
    await ElMessageBox.confirm(
      `删除会话「${session.title}」？轮次记录与对应短期记忆会一并清除。`,
      '删除会话',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return
  }
  try {
    await deleteSession(session.id)
    sessions.value = sessions.value.filter((item) => item.id !== session.id)
    if (sessionId.value === session.id) startNewSession()
    ElMessage.success('会话已删除')
  } catch (error) {
    ElMessage.error(messageOf(error))
  }
}

async function send() {
  const content = draft.value.trim()
  if (!content || sending.value) return

  const key = `local-${Date.now()}`
  turns.value.push({
    key,
    question: content,
    steps: [],
    createdAt: new Date().toISOString(),
    pending: true,
    error: null,
  })
  expandedTurns.value.add(key)
  draft.value = ''
  sending.value = true
  await scrollToBottom()

  const controller = new AbortController()
  abortController.value = controller

  try {
    await askStream(
      { content, session_id: sessionId.value },
      {
        signal: controller.signal,
        onSessionId: (id) => {
          if (id) sessionId.value = id
        },
        onStep: (step) => {
          const target = turns.value.find((item) => item.key === key)
          if (!target) return
          target.steps.push(step)
          void scrollToBottom()
        },
      },
    )

    const turn = turns.value.find((item) => item.key === key)
    if (turn) turn.pending = false

    await loadSessions()
    const matched = sessions.value.find((item) => item.id === sessionId.value)
    if (matched) {
      sessionId.value = matched.id
      sessionThreadId.value = matched.thread_id
      sessionTitle.value = matched.title
    }
  } catch (error) {
    const turn = turns.value.find((item) => item.key === key)
    if (turn) {
      turn.pending = false
      if (isAbortError(error)) {
        turn.error = '已中断本轮生成，已收到的步骤保留在会话记录中'
        void loadSessions()
      } else if (error instanceof RequestError && error.status === 404) {
        sessionId.value = null
        sessionThreadId.value = null
        sessionTitle.value = '新会话'
        turn.error = '会话不存在，已切换为新会话，请重新发送'
      } else {
        turn.error = messageOf(error)
        ElMessage.error(messageOf(error))
      }
    }
  } finally {
    sending.value = false
    abortController.value = null
    await scrollToBottom()
  }
}

function stopStreaming() {
  abortController.value?.abort()
}

function isAbortError(error: unknown): boolean {
  if (error instanceof DOMException) return error.name === 'AbortError'
  return error instanceof Error && error.name === 'AbortError'
}

function streamingHint(turn: ChatTurn): string {
  return turn.steps.length ? '等待下一步上报…' : '正在连接检索服务…'
}

function onKeydown(event: KeyboardEvent | Event) {
  const keyboardEvent = event as KeyboardEvent
  if (keyboardEvent.key !== 'Enter' || keyboardEvent.shiftKey || keyboardEvent.isComposing) return
  keyboardEvent.preventDefault()
  void send()
}

function pickAnswer(steps: Step[]): string {
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const step = steps[index]
    if (step?.answer) return step.answer
  }
  return ''
}

function pickChunks(steps: Step[]): Chunk[] {
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const step = steps[index]
    if (step?.chunks && step.chunks.length > 0) return step.chunks
  }
  return []
}

function pickFailure(steps: Step[]): string | null {
  const failed = steps.find((step) => step.status === 'error')
  return failed?.error ?? null
}

/** 不属于产品咨询时流水线直接结束，不产生 answer 与 chunks */
function isUnanswered(turn: ChatTurn): boolean {
  return (
    !turn.pending &&
    !turn.error &&
    turn.steps.length > 0 &&
    !pickAnswer(turn.steps) &&
    !pickFailure(turn.steps)
  )
}

function toggle(set: Set<string>, key: string) {
  if (set.has(key)) set.delete(key)
  else set.add(key)
}

function isTurnOpen(key: string): boolean {
  return expandedTurns.value.has(key)
}

function isChunkOpen(key: string): boolean {
  return openChunks.value.has(key)
}

async function scrollToBottom() {
  await nextTick()
  const element = bodyRef.value
  if (element) element.scrollTop = element.scrollHeight
}

onMounted(async () => {
  await loadSessions()
})
</script>

<template>
  <div class="page">
    <header class="page-header">
      <div>
        <h1 class="page-title">检索问答</h1>
        <p class="page-sub">检索、重排与答案生成在一轮内完成，答案中的 [序号] 对应下方检索片段</p>
      </div>
      <div class="page-actions">
        <el-button :loading="sessionsLoading" @click="loadSessions">
          <AppIcon name="refresh" :size="16" />
          <span class="btn-label">刷新会话</span>
        </el-button>
        <el-button type="primary" @click="startNewSession">
          <AppIcon name="plus" :size="16" />
          <span class="btn-label">新会话</span>
        </el-button>
      </div>
    </header>

    <div class="retrieval-layout">
      <aside class="session-rail surface-card">
        <div class="rail-head">
          <span class="rail-title">历史会话</span>
          <span class="rail-count num">{{ sessions.length }}</span>
        </div>

        <div class="rail-list">
          <div v-if="sessionsError" class="rail-error">
            {{ sessionsError }}
            <el-button text size="small" @click="loadSessions">重试</el-button>
          </div>
          <p v-else-if="!sessions.length && !sessionsLoading" class="rail-empty">
            还没有会话记录，发送第一个问题即可创建。
          </p>
          <div
            v-for="session in sessions"
            :key="session.id"
            class="session-item"
            :class="{ 'is-active': session.id === sessionId }"
            role="button"
            tabindex="0"
            @click="selectSession(session)"
            @keydown.enter="selectSession(session)"
          >
            <span class="session-title">{{ session.title || '未命名会话' }}</span>
            <span class="session-meta num">
              {{ session.turns }} 轮 · {{ formatDateTime(session.created_at) }}
            </span>
            <span class="session-thread mono">thread {{ session.thread_id }}</span>
            <button
              type="button"
              class="session-delete"
              title="删除会话"
              @click.stop="removeSession(session)"
            >
              <AppIcon name="trash" :size="14" />
            </button>
          </div>
        </div>
      </aside>

      <section class="conversation surface-card">
        <header class="conv-head">
          <div class="conv-heading">
            <span class="conv-title">{{ sessionTitle }}</span>
            <span class="conv-thread mono">
              {{ sessionId ? `会话 #${sessionId}` : '新会话 · 首轮发送后创建' }}
              <template v-if="sessionThreadId"> · thread {{ sessionThreadId }}</template>
            </span>
          </div>
          <el-tag size="small" type="info" effect="plain">{{ turns.length }} 轮</el-tag>
        </header>

        <div ref="bodyRef" v-loading="detailLoading" class="conv-body">
          <div v-if="!turns.length" class="conv-empty">
            <span class="empty-mark"><AppIcon name="search" :size="22" /></span>
            <p class="empty-title">从知识库开始一次检索</p>
            <p class="empty-hint">
              输入问题后会依次执行意图识别、问题改写、假设回答、联网搜索、向量检索、重排与答案生成。<br />
              Enter 发送，Shift + Enter 换行。
            </p>
          </div>

          <article v-for="(turn, index) in turns" :key="turn.key" class="turn">
            <div v-if="turn.question" class="user-row">
              <p class="user-bubble">{{ turn.question }}</p>
            </div>
            <p v-else class="history-note">
              第 {{ index + 1 }} 轮 · {{ formatDateTime(turn.createdAt) }}
            </p>

            <div v-if="turn.steps.length || turn.pending" class="answer-block">
              <div class="answer-head">
                <span class="step-count num">
                  <span v-if="turn.pending" class="pulse-dot inline-dot" />
                  {{
                    turn.pending
                      ? '正在执行检索流水线…'
                      : `${turn.steps.length} 个流水线步骤`
                  }}
                </span>
                <el-button text size="small" @click="toggle(expandedTurns, turn.key)">
                  {{ isTurnOpen(turn.key) ? '收起步骤' : '展开步骤' }}
                </el-button>
              </div>

              <ol v-if="isTurnOpen(turn.key)" class="step-list">
                <li
                  v-for="(step, stepIndex) in turn.steps"
                  :key="stepIndex"
                  class="step"
                  :class="`is-${step.status}`"
                >
                  <span class="step-dot" />
                  <span class="step-message">{{ step.message }}</span>
                  <span v-if="step.error" class="step-error">{{ step.error }}</span>
                </li>
                <li v-if="turn.pending" class="step is-running">
                  <span class="step-dot" />
                  <span class="step-message">{{ streamingHint(turn) }}</span>
                </li>
              </ol>

              <p v-if="pickAnswer(turn.steps)" class="answer-text">
                {{ pickAnswer(turn.steps) }}
              </p>
              <p v-else-if="pickFailure(turn.steps)" class="turn-error">
                {{ pickFailure(turn.steps) }}
              </p>
              <p v-else-if="isUnanswered(turn)" class="turn-note">
                本轮未产生回答：问题可能不属于产品咨询，或询问对象仍不明确。
              </p>

              <p v-if="turn.error" class="turn-error">{{ turn.error }}</p>

              <div v-if="pickChunks(turn.steps).length" class="chunks">
                <p class="chunks-title">检索片段</p>
                <div
                  v-for="(chunk, chunkIndex) in pickChunks(turn.steps)"
                  :key="chunkIndex"
                  class="chunk"
                >
                  <div class="chunk-head">
                    <span class="chunk-index num">{{ chunkIndex + 1 }}</span>
                    <span class="chunk-title">{{ chunk.title || '未命名片段' }}</span>
                    <el-tag
                      size="small"
                      :type="chunk.source === 'online' ? 'warning' : 'info'"
                      effect="light"
                    >
                      {{ chunk.source === 'online' ? '联网' : '本地' }}
                    </el-tag>
                  </div>
                  <p
                    class="chunk-content"
                    :class="{ 'is-open': isChunkOpen(`${turn.key}-${chunkIndex}`) }"
                    @click="toggle(openChunks, `${turn.key}-${chunkIndex}`)"
                  >
                    {{ chunk.content }}
                  </p>
                  <div class="chunk-foot">
                    <span v-if="chunk.filename" class="muted">{{ chunk.filename }}</span>
                    <a v-if="chunk.url" :href="chunk.url" target="_blank" rel="noopener noreferrer">
                      查看来源
                    </a>
                  </div>
                </div>
              </div>
            </div>

            <div v-else-if="turn.error" class="turn-error">{{ turn.error }}</div>
          </article>
        </div>

        <footer class="composer">
          <el-input
            v-model="draft"
            type="textarea"
            :rows="2"
            resize="none"
            placeholder="输入问题，Enter 发送，Shift + Enter 换行"
            @keydown="onKeydown"
          />
          <div class="composer-bar">
            <span class="composer-hint">同一会话共享多轮上下文，步骤与答案实时推送</span>
            <el-button v-if="sending" type="danger" plain @click="stopStreaming">
              停止生成
            </el-button>
            <el-button v-else type="primary" :disabled="!draft.trim()" @click="send">
              <AppIcon name="send" :size="15" />
              <span class="btn-label">发送</span>
            </el-button>
          </div>
        </footer>
      </section>
    </div>
  </div>
</template>

<style scoped>
.page-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}

.page-actions {
  display: flex;
  gap: 8px;
  flex: none;
}

.btn-label {
  margin-left: 6px;
}

.retrieval-layout {
  display: grid;
  grid-template-columns: 268px minmax(0, 1fr);
  gap: 16px;
  height: calc(100vh - 186px);
  min-height: 520px;
}

.session-rail {
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

.rail-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px;
  border-bottom: 1px solid var(--hairline);
}

.rail-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--ink);
}

.rail-count {
  font-size: 12px;
  color: var(--muted);
}

.rail-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.rail-empty {
  padding: 24px 12px;
  font-size: 12px;
  line-height: 1.7;
  color: var(--muted);
}

.rail-error {
  padding: 12px;
  font-size: 12px;
  color: var(--danger);
}

.session-item {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 3px;
  width: 100%;
  padding: 10px 30px 10px 10px;
  border: 0;
  border-radius: 8px;
  background: transparent;
  font: inherit;
  text-align: left;
  cursor: pointer;
  transition: background 160ms ease;
}

.session-item:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: -2px;
}

.session-item:hover {
  background: var(--surface-soft);
}

.session-item.is-active {
  background: var(--tint-lavender);
}

.session-title {
  font-size: 13px;
  font-weight: 500;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.session-meta {
  font-size: 11px;
  color: var(--muted);
}

.session-thread {
  font-size: 10px;
  color: var(--muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.session-delete {
  position: absolute;
  top: 10px;
  right: 8px;
  display: none;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 6px;
  color: var(--muted);
}

.session-item:hover .session-delete {
  display: inline-flex;
}

.session-delete:hover {
  background: #fff;
  color: var(--danger);
}

.conversation {
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

.conv-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 12px 20px;
  border-bottom: 1px solid var(--hairline);
}

.conv-heading {
  display: flex;
  flex-direction: column;
  min-width: 0;
  line-height: 1.4;
}

.conv-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--ink);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.conv-thread {
  font-size: 11px;
  color: var(--muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.conv-body {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background: var(--surface-soft);
}

.conv-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  text-align: center;
}

.empty-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 44px;
  height: 44px;
  border-radius: 10px;
  background: var(--tint-lavender);
  color: var(--accent);
}

.empty-title {
  margin-top: 12px;
  font-size: 14px;
  font-weight: 600;
  color: var(--ink);
}

.empty-hint {
  margin-top: 6px;
  font-size: 12px;
  line-height: 1.8;
  color: var(--muted);
}

.turn + .turn {
  margin-top: 20px;
}

.user-row {
  display: flex;
  justify-content: flex-end;
}

.user-bubble {
  max-width: 82%;
  padding: 10px 14px;
  border-radius: 10px 10px 2px 10px;
  background: var(--tint-lavender);
  color: var(--ink);
  font-size: 13px;
  white-space: pre-wrap;
  word-break: break-word;
}

.history-note {
  font-size: 11px;
  color: var(--muted);
}

.inline-dot {
  display: inline-block;
  margin-right: 6px;
  vertical-align: middle;
}

.pulse-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--accent);
  animation: pulse 1.4s ease-in-out infinite;
}

@keyframes pulse {
  0%,
  100% {
    opacity: 0.3;
    transform: scale(0.8);
  }
  50% {
    opacity: 1;
    transform: scale(1);
  }
}

.answer-block {
  margin-top: 10px;
  padding: 14px 16px;
  border: 1px solid var(--hairline);
  border-radius: 10px;
  background: var(--canvas);
}

.answer-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.step-count {
  font-size: 11px;
  color: var(--muted);
}

.step-list {
  margin: 10px 0 0;
  padding: 12px;
  list-style: none;
  border-radius: 8px;
  background: var(--surface-soft);
}

.step {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text);
}

.step + .step {
  margin-top: 8px;
}

.step-dot {
  width: 6px;
  height: 6px;
  flex: none;
  border-radius: 50%;
  background: var(--muted);
}

.step.is-success .step-dot {
  background: var(--success);
}

.step.is-running .step-dot {
  background: var(--accent);
}

.step.is-error .step-dot {
  background: var(--danger);
}

.step-error {
  font-size: 11px;
  color: var(--danger);
}

.answer-text {
  margin-top: 12px;
  font-size: 14px;
  line-height: 1.8;
  color: var(--ink);
  white-space: pre-wrap;
  word-break: break-word;
}

.turn-error {
  margin-top: 10px;
  padding: 10px 12px;
  border-radius: 8px;
  background: var(--tint-peach);
  color: var(--danger);
  font-size: 12px;
}

.turn-note {
  margin-top: 10px;
  padding: 10px 12px;
  border-radius: 8px;
  background: var(--surface-soft);
  color: var(--muted);
  font-size: 12px;
}

.chunks {
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid var(--hairline);
}

.chunks-title {
  margin-bottom: 8px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  color: var(--muted);
}

.chunk {
  padding: 10px 12px;
  border: 1px solid var(--hairline);
  border-radius: 8px;
}

.chunk + .chunk {
  margin-top: 8px;
}

.chunk-head {
  display: flex;
  align-items: center;
  gap: 8px;
}

.chunk-index {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  flex: none;
  border-radius: 4px;
  background: var(--tint-lavender);
  color: var(--accent);
  font-size: 11px;
  font-weight: 600;
}

.chunk-title {
  flex: 1;
  min-width: 0;
  font-size: 12px;
  font-weight: 600;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chunk-content {
  margin-top: 6px;
  font-size: 12px;
  line-height: 1.7;
  color: var(--muted);
  cursor: pointer;
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  overflow: hidden;
}

.chunk-content.is-open {
  display: block;
  -webkit-line-clamp: unset;
}

.chunk-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 6px;
  font-size: 11px;
}

.composer {
  flex: none;
  padding: 12px 20px 16px;
  border-top: 1px solid var(--hairline);
  background: var(--canvas);
}

.composer-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 8px;
}

.composer-hint {
  font-size: 11px;
  color: var(--muted);
}

@media (max-width: 1024px) {
  .retrieval-layout {
    grid-template-columns: 220px minmax(0, 1fr);
  }
}

@media (max-width: 900px) {
  .retrieval-layout {
    grid-template-columns: minmax(0, 1fr);
    height: auto;
  }

  .session-rail {
    max-height: 220px;
  }

  .conversation {
    height: 560px;
  }

  .page-header {
    flex-direction: column;
  }
}
</style>
