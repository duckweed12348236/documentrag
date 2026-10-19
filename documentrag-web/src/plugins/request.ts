import axios from 'axios'
import type { AxiosError, AxiosInstance, AxiosRequestConfig } from 'axios'

/**
 * 后端任务 / 会话 / 数据集等主键均为雪花算法生成的 64 位整数，
 * 超出 Number.MAX_SAFE_INTEGER，前端统一按字符串处理。
 */
export type SnowflakeId = string

export type RequestParams = Record<string, unknown>

const MAX_SAFE_BIGINT = BigInt(Number.MAX_SAFE_INTEGER)

const CHAR_QUOTE = 34
const CHAR_BACKSLASH = 92
const CHAR_MINUS = 45
const CHAR_PLUS = 43
const CHAR_DOT = 46
const CHAR_ZERO = 48
const CHAR_NINE = 57
const CHAR_UPPER_E = 69
const CHAR_LOWER_E = 101

function isDigit(code: number): boolean {
  return code >= CHAR_ZERO && code <= CHAR_NINE
}

function isNumberChar(code: number): boolean {
  return (
    isDigit(code) ||
    code === CHAR_PLUS ||
    code === CHAR_MINUS ||
    code === CHAR_DOT ||
    code === CHAR_UPPER_E ||
    code === CHAR_LOWER_E
  )
}

function isUnsafeIntegerLiteral(literal: string): boolean {
  if (!/^-?\d+$/.test(literal)) return false
  const value = BigInt(literal)
  return value > MAX_SAFE_BIGINT || value < -MAX_SAFE_BIGINT
}

/**
 * 给 JSON 文本中超出安全整数范围的整数字面量补上引号，
 * 这样 JSON.parse 不会把它们截断成浮点数。
 * 字符串字面量内部的内容原样保留，不会被误伤。
 */
export function quoteUnsafeIntegers(text: string): string {
  let result = ''
  let index = 0
  while (index < text.length) {
    const code = text.charCodeAt(index)
    if (code === CHAR_QUOTE) {
      let end = index + 1
      while (end < text.length) {
        const next = text.charCodeAt(end)
        if (next === CHAR_BACKSLASH) {
          end += 2
          continue
        }
        end += 1
        if (next === CHAR_QUOTE) break
      }
      result += text.slice(index, end)
      index = end
      continue
    }
    if (code === CHAR_MINUS || isDigit(code)) {
      let end = index + 1
      while (end < text.length && isNumberChar(text.charCodeAt(end))) end += 1
      const literal = text.slice(index, end)
      result += isUnsafeIntegerLiteral(literal) ? `"${literal}"` : literal
      index = end
      continue
    }
    result += text.charAt(index)
    index += 1
  }
  return result
}

/**
 * 无损解析 JSON：大整数（雪花 ID）会被解析成字符串，其余字段保持原类型。
 */
export function parseJSON<T = unknown>(text: string): T {
  return JSON.parse(quoteUnsafeIntegers(text)) as T
}

export interface ApiErrorBody {
  detail?: unknown
}

/** 业务错误 detail 是字符串；422 校验错误 detail 是字段错误数组 */
function detailMessage(detail: unknown): string | null {
  if (typeof detail === 'string' && detail.trim()) return detail
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (item && typeof item === 'object' && 'msg' in item) {
          return String((item as { msg: unknown }).msg)
        }
        return null
      })
      .filter((message): message is string => Boolean(message))
    if (messages.length) return messages.join('；')
  }
  return null
}

const STATUS_MESSAGE: Record<number, string> = {
  400: '请求参数有误',
  404: '资源不存在',
  409: '当前状态不允许该操作',
  422: '请求参数校验失败',
  500: '服务端异常',
  502: '对象存储不可用',
  503: '服务暂不可用，请稍后重试',
}

export class RequestError extends Error {
  readonly status: number | null
  readonly data: unknown

  constructor(message: string, status: number | null = null, data: unknown = null) {
    super(message)
    this.name = 'RequestError'
    this.status = status
    this.data = data
  }
}

function toRequestError(error: AxiosError<ApiErrorBody>): RequestError {
  if (error.response) {
    const { status, data } = error.response
    const detail = data && typeof data === 'object' ? data.detail : undefined
    const message = detailMessage(detail) || STATUS_MESSAGE[status] || `请求失败（HTTP ${status}）`
    return new RequestError(message, status, data)
  }
  if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
    return new RequestError('请求超时，请稍后重试')
  }
  return new RequestError('网络异常：后端未响应，可能是跨域被拦截或服务未启动')
}

/**
 * 默认走同源 `/api`，由 Vite 开发 / 预览代理转发到后端，避免浏览器跨域。
 * 部署时可通过 `VITE_API_BASE_URL` 指向真实网关。
 */
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

const http: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60_000,
  transformResponse: [
    (data: unknown) => {
      if (typeof data !== 'string') return data
      const text = data.trim()
      if (text === '') return data
      const first = text.charCodeAt(0)
      if (first !== 123 && first !== 91) return data
      try {
        return parseJSON(text)
      } catch {
        return data
      }
    },
  ],
})

http.interceptors.response.use(
  (response) => response.data,
  (error: AxiosError<ApiErrorBody>) => Promise.reject(toRequestError(error)),
)

/** 拼接路径参数：保留 `/` 分隔，其余字符（中文、空格等）按 URL 编码 */
export function encodePath(value: string): string {
  return value
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/')
}

export function get<T>(url: string, params?: RequestParams, config?: AxiosRequestConfig): Promise<T> {
  return http.get(url, { ...config, params }) as unknown as Promise<T>
}

export function post<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
  return http.post(url, data, config) as unknown as Promise<T>
}

export function put<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
  return http.put(url, data, config) as unknown as Promise<T>
}

export function patch<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
  return http.patch(url, data, config) as unknown as Promise<T>
}

export function del<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
  return http.delete(url, config) as unknown as Promise<T>
}

/** multipart 上传，由浏览器自动补全 boundary，不要手动设置 Content-Type */
export function upload<T>(url: string, file: File, config?: AxiosRequestConfig): Promise<T> {
  const form = new FormData()
  form.append('file', file)
  return post<T>(url, form, config)
}

export const request = { get, post, put, patch, delete: del, upload, parseJSON, encodePath }

export default request
