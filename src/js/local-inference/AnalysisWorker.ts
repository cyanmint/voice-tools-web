import {buildLocalAnalysis} from './analysis.js'
import {inferAudio, loadLocalModels} from './LocalInference'
import curves from '@/data/vox1_kde_curves.json'

interface RequestMessage {
  id: number
  samples: ArrayBuffer
  sampleRate: number
  duration: number
  inferenceEnabled: boolean
  baseUrl: string
}

const workerScope = self as unknown as {
  addEventListener(type: 'message', listener: (event: MessageEvent<RequestMessage>) => void): void
  postMessage(message: unknown, transfer?: Transferable[]): void
}

function resample(samples: Float32Array, sourceRate: number, targetRate: number): Float32Array {
  if (sourceRate === targetRate) return samples
  const length = Math.max(1, Math.round(samples.length * targetRate / sourceRate))
  const output = new Float32Array(length)
  const scale = sourceRate / targetRate
  for (let index = 0; index < length; index += 1) {
    const position = index * scale
    const left = Math.min(samples.length - 1, Math.floor(position))
    const right = Math.min(samples.length - 1, left + 1)
    const fraction = position - left
    output[index] = samples[left] * (1 - fraction) + samples[right] * fraction
  }
  return output
}

workerScope.addEventListener('message', async (event: MessageEvent<RequestMessage>) => {
  const request = event.data
  try {
    const samples = new Float32Array(request.samples)
    let infer: ((pcm: Float32Array, duration: number) => Promise<unknown>) | undefined
    if (request.inferenceEnabled) {
      workerScope.postMessage({id: request.id, type: 'progress', message: 'Loading local TensorFlow.js models…'})
      await loadLocalModels(request.baseUrl)
      infer = async (pcm, duration) => {
        const modelPcm = resample(pcm, request.sampleRate, 16000)
        const output = await inferAudio(modelPcm, message => workerScope.postMessage({id: request.id, type: 'progress', message}))
        return {...output, duration}
      }
    }

    const result = await buildLocalAnalysis(samples, request.sampleRate, request.duration, {
      curves,
      infer,
      onProgress: message => workerScope.postMessage({id: request.id, type: 'progress', message}),
    })
    const transfer: Transferable[] = result.spec.map(row => row.buffer)
    for (const values of Object.values(result.freqArrays)) transfer.push(values.buffer)
    workerScope.postMessage({id: request.id, type: 'result', result}, transfer)
  } catch (error) {
    workerScope.postMessage({id: request.id, type: 'error', message: error instanceof Error ? error.message : String(error)})
  }
})
