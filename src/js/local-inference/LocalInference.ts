import { resolveModelUrls } from './model-urls.js'
import { extractMelFeatures, makeFeatureWindows } from './audio-features.js'
import { diagonalTransitionLog, viterbiDecode } from './segmentation.js'

type Label = 'noEnergy' | 'speech' | 'music' | 'noise' | 'female' | 'male'
export type MLFrame = [Label, number, number, number | null]

type Runtime = { tf: any, backend: 'webgpu' | 'cpu' }
type LoadedModels = { vad: any, gender: any, runtime: Runtime }

const MODEL_WINDOW = 68
const ACTIVITY_STRIDE = 2
const SAMPLE_RATE = 16000
const TF_LABELS: Label[] = ['speech', 'music', 'noise']
const GENDER_LABELS: Label[] = ['female', 'male']
let runtimePromise: Promise<Runtime> | null = null
let models: LoadedModels | null = null

async function getRuntime(): Promise<Runtime> {
  if (runtimePromise) return runtimePromise
  runtimePromise = (async () => {
    const tf = await import('@tensorflow/tfjs')
    try { await import('@tensorflow/tfjs-backend-webgpu') } catch { /* unsupported browser/build */ }

    const gpu = (navigator as any).gpu
    if (gpu) {
      try {
        const adapter = await gpu.requestAdapter()
        if (adapter && await tf.setBackend('webgpu')) {
          await tf.ready()
          return { tf, backend: 'webgpu' as const }
        }
      } catch (error) {
        console.info('WebGPU unavailable; using CPU.', error)
      }
    }
    if (!await tf.setBackend('cpu')) throw new Error('TensorFlow.js CPU backend could not be initialized')
    await tf.ready()
    return { tf, backend: 'cpu' as const }
  })()
  try { return await runtimePromise } catch (error) {
    runtimePromise = null
    throw error
  }
}

async function loadModel(url: string, role: 'VAD' | 'gender', tf: any): Promise<any> {
  const loaded = await tf.loadLayersModel(url)
  try {
    validateModel(loaded, role)
    return loaded
  } catch (error) {
    loaded.dispose()
    throw error
  }
}

function validateModel(loaded: any, role: 'VAD' | 'gender'): void {
  const shape = loaded.inputs?.[0]?.shape
  const outputShape = loaded.outputs?.[0]?.shape
  const classes = outputShape?.[outputShape.length - 1]
  if (!shape || shape.length !== 3 || shape[1] !== MODEL_WINDOW || (shape[2] !== 21 && shape[2] !== 24)) {
    throw new Error(`${role}: expected model input [batch, 68, 21 or 24], got ${JSON.stringify(shape)}`)
  }
  const expectedClasses = role === 'VAD' ? TF_LABELS.length : GENDER_LABELS.length
  if (classes !== expectedClasses) throw new Error(`${role}: expected ${expectedClasses} output classes, got ${classes}`)
}

function replaceModels(vad: any, gender: any, runtime: Runtime): void {
  const previous = models
  models = { vad, gender, runtime }
  previous?.vad.dispose()
  previous?.gender.dispose()
}

export async function loadLocalModels(baseUrl = import.meta.env.BASE_URL): Promise<string> {
  const runtime = await getRuntime()
  const {vad: vadUrl, gender: genderUrl} = resolveModelUrls(baseUrl, globalThis.location.origin)
  const vad = await loadModel(vadUrl, 'VAD', runtime.tf)
  try {
    const gender = await loadModel(genderUrl, 'gender', runtime.tf)
    replaceModels(vad, gender, runtime)
  } catch (error) {
    vad.dispose()
    throw error
  }
  return runtime.backend
}

export function currentBackend(): string | null {
  return models?.runtime.backend ?? null
}

async function predict(model: any, windows: number[][][], tf: any, onProgress?: (fraction: number) => void): Promise<number[][]> {
  const batchSize = 32
  const classes = model.outputs[0].shape.at(-1)
  const scores: number[][] = []
  for (let start = 0; start < windows.length; start += batchSize) {
    const batch = windows.slice(start, start + batchSize)
    const input = tf.tensor3d(batch, [batch.length, MODEL_WINDOW, model.inputs[0].shape[2]])
    let output: any
    try {
      output = model.predict(input)
      const values = await output.data()
      for (let row = 0; row < batch.length; row += 1) {
        const current = Array.from(values.slice(row * classes, (row + 1) * classes), Number)
        if (!current.every(Number.isFinite)) throw new Error('Model returned a non-finite probability')
        scores.push(current)
      }
    } finally {
      input.dispose()
      if (output) output.dispose()
    }
    onProgress?.(Math.min(1, (start + batch.length) / windows.length))
  }
  return scores
}

function appendRunLengthEncoded(frames: MLFrame[], label: Label, start: number, end: number, confidence: number | null): void {
  if (end <= start) return
  const previous = frames.at(-1)
  if (previous && previous[0] === label && Math.abs(previous[2] - start) < 1e-7) {
    const previousLength = previous[2] - previous[1]
    const nextLength = end - start
    if (previous[3] !== null && confidence !== null) {
      previous[3] = (previous[3] * previousLength + confidence * nextLength) / (previousLength + nextLength)
    } else previous[3] = null
    previous[2] = end
  } else frames.push([label, start, end, confidence])
}

function decodeMono(buffer: AudioBuffer): Float32Array {
  const output = new Float32Array(buffer.length)
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    const data = buffer.getChannelData(channel)
    for (let i = 0; i < data.length; i += 1) output[i] += data[i] / buffer.numberOfChannels
  }
  return output
}

export async function inferAudio(buffer: AudioBuffer | Float32Array, onProgress?: (message: string) => void): Promise<{ frames: MLFrame[], backend: string }> {
  if (!models) throw new Error('Download the local TensorFlow.js models before analyzing audio')
  const pcm = buffer instanceof Float32Array ? buffer : decodeMono(buffer)
  const duration = buffer instanceof Float32Array ? pcm.length / SAMPLE_RATE : buffer.duration
  if (!(buffer instanceof Float32Array) && buffer.sampleRate !== SAMPLE_RATE) throw new Error(`Expected 16 kHz decoded audio, got ${buffer.sampleRate} Hz`)
  const { tf, backend } = models.runtime
  const features = extractMelFeatures(pcm, { sampleRate: SAMPLE_RATE })
  if (features.mel.length === 0) throw new Error('Audio is too short to extract model features')

  const frameCount = Math.min(Math.ceil(duration / 0.02), Math.ceil(features.logEnergy.length / ACTIVITY_STRIDE))
  const energyMean = features.logEnergy.reduce((sum, value) => sum + value, 0) / features.logEnergy.length
  const energyThreshold = energyMean + Math.log(0.03)
  const activityProbabilities = Array.from({ length: frameCount }, (_, frame) => {
    const active = features.logEnergy[Math.min(frame * ACTIVITY_STRIDE, features.logEnergy.length - 1)] > energyThreshold
    return active ? [1e-8, 1 - 1e-8] : [1 - 1e-8, 1e-8]
  })
  const activityPath = viterbiDecode(activityProbabilities, diagonalTransitionLog(2, 150))

  let mel = features.mel
  if (mel.length < MODEL_WINDOW) {
    let minimum = Infinity
    for (const row of mel) for (const value of row) if (value < minimum) minimum = value
    const padding = Array.from({ length: MODEL_WINDOW - mel.length }, () => Array(24).fill(minimum))
    mel = [...mel, ...padding]
  }
  const vadWindows = makeFeatureWindows(mel, { featureCount: models.vad.inputs[0].shape[2] }).windows
  const vadScores = await predict(models.vad, vadWindows, tf, fraction => onProgress?.(`语音/音乐/噪声模型 ${Math.round(fraction * 100)}%`))
  const frameScores = vadScores.slice(0, frameCount)
  if (frameScores.length !== frameCount) throw new Error('VAD model output is shorter than the decoded audio')

  const labelIndices = Array<Label>(frameCount).fill('noEnergy')
  const confidence = Array<number | null>(frameCount).fill(null)
  const vadTransition = diagonalTransitionLog(TF_LABELS.length, 80)
  let rangeStart = -1
  const speechRanges: Array<[number, number]> = []
  for (let frame = 0; frame <= frameCount; frame += 1) {
    const active = frame < frameCount && activityPath[frame] === 1
    if (active && rangeStart < 0) rangeStart = frame
    if (!active && rangeStart >= 0) {
      const end = frame
      const path = viterbiDecode(frameScores.slice(rangeStart, end), vadTransition)
      for (let offset = 0; offset < path.length; offset += 1) {
        const index = rangeStart + offset
        const classIndex = path[offset]
        labelIndices[index] = TF_LABELS[classIndex]
        confidence[index] = frameScores[index][classIndex]
        if (TF_LABELS[classIndex] === 'speech') {
          const last = speechRanges.at(-1)
          if (last && last[1] === index) last[1] = index + 1
          else speechRanges.push([index, index + 1])
        }
      }
      rangeStart = -1
    }
  }

  const genderWindows = makeFeatureWindows(mel, { featureCount: models.gender.inputs[0].shape[2] }).windows
  const genderFrameCount = Math.min(frameCount, genderWindows.length)
  const genderScores = await predict(models.gender, genderWindows, tf, fraction => onProgress?.(`性别模型 ${Math.round(fraction * 100)}%`))
  const genderTransition = diagonalTransitionLog(GENDER_LABELS.length, 80)
  for (const [start, end] of speechRanges) {
    const stop = Math.min(end, genderFrameCount)
    if (stop <= start) continue
    const scores = genderScores.slice(start, stop)
    const path = viterbiDecode(scores, genderTransition)
    for (let offset = 0; offset < path.length; offset += 1) {
      const index = start + offset
      const classIndex = path[offset]
      labelIndices[index] = GENDER_LABELS[classIndex]
      confidence[index] = scores[offset][classIndex]
    }
  }

  const output: MLFrame[] = []
  const step = 0.02
  for (let frame = 0; frame < frameCount; frame += 1) {
    const start = frame * step
    const end = Math.min(duration, (frame + 1) * step)
    appendRunLengthEncoded(output, labelIndices[frame], start, end, confidence[frame])
  }
  return { frames: output, backend }
}
