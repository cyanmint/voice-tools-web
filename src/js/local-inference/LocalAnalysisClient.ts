import type {LocalAnalysisResult} from './LocalResultStore'

interface PendingRequest {
  resolve: (value: LocalAnalysisResult) => void
  reject: (error: Error) => void
  onProgress: (message: string) => void
}

interface WorkerMessage {
  id: number
  type: 'progress' | 'result' | 'error'
  message?: string
  result?: LocalAnalysisResult
}

let worker: Worker | null = null
let nextId = 1
const pending = new Map<number, PendingRequest>()

function getWorker(): Worker {
  if (worker) return worker
  worker = new Worker(new URL('./AnalysisWorker.ts', import.meta.url), {type: 'module'})
  worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
    const request = pending.get(event.data.id)
    if (!request) return
    if (event.data.type === 'progress') request.onProgress(event.data.message || 'Analyzing locally…')
    else if (event.data.type === 'result' && event.data.result) {
      pending.delete(event.data.id)
      request.resolve(event.data.result)
    } else if (event.data.type === 'error') {
      pending.delete(event.data.id)
      request.reject(new Error(event.data.message || 'Local analysis failed'))
    }
  }
  worker.onerror = event => {
    const error = new Error(event.message || 'Local analysis worker failed')
    for (const request of pending.values()) request.reject(error)
    pending.clear()
    worker?.terminate()
    worker = null
  }
  return worker
}

export function runLocalAnalysis(
  samples: Float32Array,
  sampleRate: number,
  duration: number,
  inferenceEnabled: boolean,
  onProgress: (message: string) => void,
): Promise<LocalAnalysisResult> {
  const id = nextId++
  const pcm = samples.slice()
  return new Promise((resolve, reject) => {
    pending.set(id, {resolve, reject, onProgress})
    try {
      getWorker().postMessage({
        id,
        samples: pcm.buffer,
        sampleRate,
        duration,
        inferenceEnabled,
        baseUrl: import.meta.env.BASE_URL,
      }, [pcm.buffer])
    } catch (error) {
      pending.delete(id)
      reject(error instanceof Error ? error : new Error(String(error)))
    }
  })
}
