const MODEL_CACHE = 'voice-tools-models-v1'
const MODEL_FILES = [
  'models/tfjs/keras_speech_music_noise_cnn/model.json',
  'models/tfjs/keras_speech_music_noise_cnn/group1-shard1of1.bin',
  'models/tfjs/keras_male_female_cnn/model.json',
  'models/tfjs/keras_male_female_cnn/group1-shard1of1.bin',
]

function publicUrl(path: string): string {
  return new URL(`${import.meta.env.BASE_URL}${path}`, globalThis.location.origin).href
}

export function requiredModelUrls(): string[] {
  return MODEL_FILES.map(publicUrl)
}

export async function modelsDownloaded(): Promise<boolean> {
  if (!('caches' in globalThis)) return false
  const cache = await caches.open(MODEL_CACHE)
  const urls = requiredModelUrls()
  const matches = await Promise.all(urls.map(url => cache.match(url)))
  return matches.every(Boolean)
}

export async function downloadModels(onProgress: (message: string) => void = () => undefined): Promise<void> {
  if (!('serviceWorker' in navigator) || !('MessageChannel' in globalThis)) {
    throw new Error('This browser does not support offline model downloads via Service Worker.')
  }
  const registration = await navigator.serviceWorker.ready
  const worker = navigator.serviceWorker.controller || registration.active
  if (!worker) throw new Error('The local model service is not active yet. Reload the page and try again.')

  await new Promise<void>((resolve, reject) => {
    const channel = new MessageChannel()
    const timeout = globalThis.setTimeout(() => {
      channel.port1.close()
      reject(new Error('Model download timed out. Check your connection and try again.'))
    }, 10 * 60 * 1000)
    channel.port1.onmessage = event => {
      const message = event.data
      if (message?.type === 'progress') onProgress(message.message)
      if (message?.type === 'complete') {
        globalThis.clearTimeout(timeout)
        channel.port1.close()
        resolve()
      }
      if (message?.type === 'error') {
        globalThis.clearTimeout(timeout)
        channel.port1.close()
        reject(new Error(message.message || 'Model download failed.'))
      }
    }
    worker.postMessage({ type: 'DOWNLOAD_MODELS', urls: requiredModelUrls(), cacheName: MODEL_CACHE }, [channel.port2])
  })
}

export async function registerLocalModelService(): Promise<void> {
  if (!('serviceWorker' in navigator)) return
  const scriptUrl = `${import.meta.env.BASE_URL}service-worker.js`
  await navigator.serviceWorker.register(scriptUrl, { scope: import.meta.env.BASE_URL })
}
