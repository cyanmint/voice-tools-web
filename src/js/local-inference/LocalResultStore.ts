export interface LocalAnalysisResult {
  result: {
    means: Record<string, number | null>
    fem_prob: Record<string, number | null>
  }
  ml: Array<[string, number, number, number | null]>
  ml_enabled: boolean
  inference_backend: string | null
  ml_error: string | null
  spec: Float32Array[]
  spec_sr: number
  audio_dur: number
  freqArrays: Record<string, Float32Array>
}

const DB_NAME = 'voice-tools-local-results'
const STORE_NAME = 'analyses'
const DB_VERSION = 1

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE_NAME)) request.result.createObjectStore(STORE_NAME)
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error || new Error('Could not open local analysis storage'))
    request.onblocked = () => reject(new Error('Local analysis storage is blocked by another tab'))
  })
}

export async function saveLocalAnalysis(id: string, result: LocalAnalysisResult): Promise<void> {
  const database = await openDatabase()
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, 'readwrite')
      transaction.objectStore(STORE_NAME).put(result, id)
      transaction.oncomplete = () => resolve()
      transaction.onerror = () => reject(transaction.error || new Error('Could not save local analysis'))
      transaction.onabort = () => reject(transaction.error || new Error('Local analysis save was aborted'))
    })
  } finally {
    database.close()
  }
}

export async function getLocalAnalysis(id: string): Promise<LocalAnalysisResult | null> {
  const database = await openDatabase()
  try {
    return await new Promise((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, 'readonly')
      const request = transaction.objectStore(STORE_NAME).get(id)
      request.onsuccess = () => resolve(request.result || null)
      request.onerror = () => reject(request.error || new Error('Could not read local analysis'))
    })
  } finally {
    database.close()
  }
}
