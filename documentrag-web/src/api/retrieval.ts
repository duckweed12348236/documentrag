import { API_BASE_URL, RequestError, del, get, parseJSON } from '@/plugins/request'
import type { SnowflakeId } from '@/plugins/request'
import { readSSE } from './sse'
import type { SessionDetail, SessionSummary, Step } from './types'

export interface AskPayload {
  content: string
  /** 延续已有会话时传会话主键；首轮传 null，由服务端新建并在响应头返回 */
  session_id: SnowflakeId | null
}

export interface AskResult {
  steps: Step[]
  sessionId: SnowflakeId | null
}

export interface AskStreamOptions {
  signal?: AbortSignal
  onStep?: (step: Step) => void
  onSessionId?: (sessionId: SnowflakeId | null) => void
}

/**
 * 检索问答为 SSE 流式接口：每收到一帧就回调 onStep，整轮结束后返回全部步骤。
 * 会话主键从响应头 X-Session-Id 读取，可通过 onSessionId 立即拿到。
 * 传 AbortSignal 可中断生成，中断会抛出 name 为 AbortError 的异常。
 */
export async function askStream(
  payload: AskPayload,
  options: AskStreamOptions = {},
): Promise<AskResult> {
  const response = await fetch(`${API_BASE_URL}/retrieval`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: serializeAskPayload(payload),
    signal: options.signal,
  })

  if (!response.ok) {
    throw new RequestError(await readErrorMessage(response), response.status)
  }

  const sessionId = response.headers.get('X-Session-Id')
  options.onSessionId?.(sessionId)

  if (!response.body) {
    throw new RequestError('当前浏览器不支持流式响应')
  }

  const steps: Step[] = []
  for await (const event of readSSE(response.body)) {
    if (event.event === 'error') {
      throw new RequestError(parseStreamError(event.data), null, event.data)
    }
    if (event.data === '[DONE]') break
    if (!event.data) continue

    let step: Step
    try {
      step = parseJSON<Step>(event.data)
    } catch {
      continue
    }
    steps.push(step)
    options.onStep?.(step)
  }

  return { steps, sessionId }
}

/**
 * session_id 是 64 位整数，直接用 JSON.stringify 会丢精度，
 * 这里对纯数字 ID 原样写入，保证服务端拿到精确值。
 */
function serializeAskPayload(payload: AskPayload): string {
  const content = JSON.stringify(payload.content)
  const sessionId = payload.session_id
  if (sessionId === null) return `{"content":${content},"session_id":null}`
  if (/^\d+$/.test(sessionId)) return `{"content":${content},"session_id":${sessionId}}`
  return JSON.stringify({ content: payload.content, session_id: sessionId })
}

async function readErrorMessage(response: Response): Promise<string> {
  try {
    const data = (await response.json()) as { detail?: unknown }
    if (typeof data.detail === 'string' && data.detail.trim()) return data.detail
    if (Array.isArray(data.detail)) {
      const messages = data.detail
        .map((item) => {
          if (item && typeof item === 'object' && 'msg' in item) {
            return String((item as { msg: unknown }).msg)
          }
          return null
        })
        .filter((message): message is string => Boolean(message))
      if (messages.length) return messages.join('；')
    }
  } catch {
    // 非 JSON 错误响应，使用状态码兜底
  }
  return `请求失败（HTTP ${response.status}）`
}

function parseStreamError(data: string): string {
  try {
    const parsed = JSON.parse(data) as { error?: string }
    if (parsed?.error) return parsed.error
  } catch {
    // 保持原始文本
  }
  return data || '检索失败'
}

export function fetchSessions(limit?: number): Promise<SessionSummary[]> {
  return get<SessionSummary[]>('/retrieval/session', { limit })
}

export function fetchSession(sessionId: string): Promise<SessionDetail> {
  return get<SessionDetail>(`/retrieval/session/${encodeURIComponent(sessionId)}`)
}

export function deleteSession(sessionId: string): Promise<void> {
  return del<void>(`/retrieval/session/${encodeURIComponent(sessionId)}`)
}
