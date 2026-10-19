import fs from 'node:fs'
import path from 'node:path'
import type { IncomingMessage, ServerResponse } from 'node:http'

import { fileURLToPath, URL } from 'node:url'

import { defineConfig, loadEnv } from 'vite'
import type { Plugin } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

/**
 * 本地开发 / 预览时把后端 storage 目录里的 PDF 暴露为 /local-storage/<文件名>，
 * 用于在浏览器中预览尚未同步到对象存储的文档。
 */
function localPdfPlugin(storageDir: string): Plugin {
  const serve = (req: IncomingMessage, res: ServerResponse, next: (error?: unknown) => void) => {
    let fileName = ''
    try {
      fileName = path.basename(decodeURIComponent((req.url ?? '').split('?')[0] ?? ''))
    } catch {
      next()
      return
    }
    if (!fileName.toLowerCase().endsWith('.pdf')) {
      next()
      return
    }

    const safeDir = path.resolve(storageDir)
    const filePath = path.join(safeDir, fileName)
    if (!path.resolve(filePath).startsWith(safeDir)) {
      next()
      return
    }

    fs.stat(filePath, (error, stat) => {
      if (error || !stat.isFile()) {
        next()
        return
      }

      res.setHeader('Content-Type', 'application/pdf')
      res.setHeader('Accept-Ranges', 'bytes')
      res.setHeader('Cache-Control', 'no-cache')

      const range = req.headers.range
      const match = range ? /^bytes=(\d*)-(\d*)$/.exec(range) : null
      if (match) {
        const startText = match[1] ?? ''
        const endText = match[2] ?? ''
        let start = startText ? Number(startText) : 0
        let end = endText ? Number(endText) : stat.size - 1

        if (!startText && endText) {
          start = Math.max(0, stat.size - Number(endText))
          end = stat.size - 1
        }
        if (!Number.isFinite(start) || start >= stat.size) {
          res.statusCode = 416
          res.setHeader('Content-Range', `bytes */${stat.size}`)
          res.end()
          return
        }

        end = Math.min(end, stat.size - 1)
        res.statusCode = 206
        res.setHeader('Content-Range', `bytes ${start}-${end}/${stat.size}`)
        res.setHeader('Content-Length', end - start + 1)
        if (req.method === 'HEAD') {
          res.end()
          return
        }
        fs.createReadStream(filePath, { start, end }).pipe(res)
        return
      }

      res.statusCode = 200
      res.setHeader('Content-Length', stat.size)
      if (req.method === 'HEAD') {
        res.end()
        return
      }
      fs.createReadStream(filePath).pipe(res)
    })
  }

  return {
    name: 'local-storage-pdf',
    configureServer(server) {
      server.middlewares.use('/local-storage', serve)
    },
    configurePreviewServer(server) {
      server.middlewares.use('/local-storage', serve)
    },
  }
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiTarget = env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000'
  const storageDir =
    env.VITE_LOCAL_STORAGE_DIR || path.resolve(process.cwd(), '../productrag-backend/storage')

  const apiProxy = {
    target: apiTarget,
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/api/, ''),
  }

  return {
    plugins: [
      vue(),
      vueDevTools(),
      localPdfPlugin(storageDir),
    ],
    server: {
      proxy: { '/api': apiProxy },
    },
    preview: {
      proxy: { '/api': apiProxy },
    },
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url)),
      },
    },
  }
})
