export interface SSEEvent {
  event: string
  data: string
}

const FRAME_SEPARATOR = /\r?\n\r?\n/

function parseFrame(frame: string): SSEEvent | null {
  let event = 'message'
  const dataLines: string[] = []
  let hasField = false

  for (const line of frame.split(/\r?\n/)) {
    if (!line || line.startsWith(':')) continue
    const colon = line.indexOf(':')
    const field = colon === -1 ? line : line.slice(0, colon)
    let value = colon === -1 ? '' : line.slice(colon + 1)
    if (value.startsWith(' ')) value = value.slice(1)

    if (field === 'event') {
      event = value
      hasField = true
    } else if (field === 'data') {
      dataLines.push(value)
      hasField = true
    }
  }

  if (!hasField) return null
  return { event, data: dataLines.join('\n') }
}

/** 逐块解析 SSE 文本，处理事件被拆到多个网络分片、以及 CRLF 分隔的情况 */
export class SSEDecoder {
  private buffer = ''

  push(chunk: string): SSEEvent[] {
    this.buffer += chunk
    const events: SSEEvent[] = []
    let match = FRAME_SEPARATOR.exec(this.buffer)
    while (match) {
      const frame = this.buffer.slice(0, match.index)
      this.buffer = this.buffer.slice(match.index + match[0].length)
      const event = parseFrame(frame)
      if (event) events.push(event)
      match = FRAME_SEPARATOR.exec(this.buffer)
    }
    return events
  }

  flush(): SSEEvent[] {
    const rest = this.buffer.trim()
    this.buffer = ''
    if (!rest) return []
    const event = parseFrame(rest)
    return event ? [event] : []
  }
}

export async function* readSSE(stream: ReadableStream<Uint8Array>): AsyncGenerator<SSEEvent> {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  const sse = new SSEDecoder()

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      const text = decoder.decode(value, { stream: true })
      for (const event of sse.push(text)) yield event
    }
    for (const event of sse.push(decoder.decode())) yield event
    for (const event of sse.flush()) yield event
  } finally {
    reader.releaseLock()
  }
}
