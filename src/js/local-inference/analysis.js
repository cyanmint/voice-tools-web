import { calculatePraatFrequencyArray, calculatePraatSpectralTilt } from './praat-analysis.js'
import { computeMelSpectrogram } from './signal-analysis.js'
import vox1Data from '../../data/vox1_data.js'

const FEATURES = ['pitch', 'f1', 'f2', 'f3', 'tilt']

function pairwiseFloat32Sum(values, start, end) {
  const length = end - start
  const add = (left, right) => Math.fround(left + right)
  if (length < 8) {
    let sum = -0
    for (let index = start; index < end; index += 1) sum = add(sum, Math.fround(values[index]))
    return sum
  }
  if (length <= 128) {
    const accumulators = Array.from({ length: 8 }, (_, offset) => Math.fround(values[start + offset]))
    let index = start + 8
    const alignedEnd = end - (length % 8)
    for (; index < alignedEnd; index += 8) {
      for (let offset = 0; offset < 8; offset += 1) {
        accumulators[offset] = add(accumulators[offset], Math.fround(values[index + offset]))
      }
    }
    let sum = add(add(add(accumulators[0], accumulators[1]), add(accumulators[2], accumulators[3])),
      add(add(accumulators[4], accumulators[5]), add(accumulators[6], accumulators[7])))
    for (; index < end; index += 1) sum = add(sum, Math.fround(values[index]))
    return sum
  }
  let midpoint = Math.floor(length / 2)
  midpoint -= midpoint % 8
  const split = start + midpoint
  return add(pairwiseFloat32Sum(values, start, split), pairwiseFloat32Sum(values, split, end))
}

export function meanNumpyFloat32(values) {
  const finite = values.filter(Number.isFinite)
  if (!finite.length) return null
  return Math.fround(pairwiseFloat32Sum(finite, 0, finite.length) / finite.length)
}

function meanFinite(values) {
  return meanNumpyFloat32(values)
}

function percentile(values, probability) {
  const sorted = [...values].sort((left, right) => left - right)
  const position = (sorted.length - 1) * probability
  const lower = Math.floor(position)
  const upper = Math.ceil(position)
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower)
}

function gaussianKde(values, value) {
  const count = values.length
  const mean = values.reduce((sum, sample) => sum + sample, 0) / count
  const variance = values.reduce((sum, sample) => sum + (sample - mean) ** 2, 0) / (count - 1)
  const bandwidth = Math.sqrt(variance) * count ** -0.2
  const normalizer = count * bandwidth * Math.sqrt(2 * Math.PI)
  return values.reduce((sum, sample) => {
    const standardized = (value - sample) / bandwidth
    return sum + Math.exp(-0.5 * standardized * standardized) / normalizer
  }, 0)
}

export function feminineProbability(distributions, feature, value) {
  if (!Number.isFinite(value)) return null
  const groups = distributions?.[feature]
  const female = groups?.f
  const male = groups?.m
  if (!Array.isArray(female) || !Array.isArray(male) || female.length < 2 || male.length < 2) return null

  if (value < percentile(male, 0.01)) return 0
  if (value > percentile(female, 0.99)) return 1
  const femaleDensity = gaussianKde(female, value)
  const maleDensity = gaussianKde(male, value)
  const total = femaleDensity + maleDensity
  return total > 0 ? femaleDensity / total : null
}

/** Build the browser-local equivalent of the fields consumed by Results.vue. */
export async function buildLocalAnalysis(samples, sampleRate, duration, { infer, curves, onProgress } = {}) {
  if (!(samples instanceof Float32Array) || !samples.length) throw new TypeError('Audio must decode to non-empty Float32 PCM')
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) throw new RangeError('Audio sample rate must be positive')
  if (!Number.isFinite(duration) || duration <= 0) throw new RangeError('Audio duration must be positive')
  if (!curves) throw new Error('Gender reference curves are unavailable')

  onProgress?.('Computing local voice statistics…')
  const frequencyRows = calculatePraatFrequencyArray(samples, sampleRate)
  const means = Object.fromEntries(FEATURES.slice(0, 4).map((feature, column) => [
    feature, meanFinite(frequencyRows.map(row => row[column])),
  ]))
  means.tilt = calculatePraatSpectralTilt(samples)
  const fem_prob = Object.fromEntries(FEATURES.map(feature => [feature, feminineProbability(vox1Data, feature, means[feature])]))
  const freqArrays = Object.fromEntries(FEATURES.slice(0, 4).map((feature, column) => [
    feature, Float32Array.from(frequencyRows, row => Number.isFinite(row[column]) ? row[column] : Number.NaN),
  ]))

  onProgress?.('Computing local spectrogram…')
  const spec = computeMelSpectrogram(samples, sampleRate).map(row => Float32Array.from(row))
  let ml = []
  let inference_backend = null
  let ml_error = null
  if (infer) {
    try {
      const output = await infer(samples, duration)
      if (Array.isArray(output)) ml = output
      else {
        ml = output?.frames || []
        inference_backend = output?.backend || null
      }
    } catch (error) {
      ml_error = error instanceof Error ? error.message : String(error)
    }
  }
  return {
    result: { means, fem_prob },
    ml,
    ml_enabled: Boolean(infer),
    inference_backend,
    ml_error,
    spec,
    spec_sr: sampleRate,
    audio_dur: duration,
    freqArrays,
  }
}
