import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { buildLocalAnalysis, meanNumpyFloat32 } from '../src/js/local-inference/analysis.js'
import { calculatePraatFrequencyArray, praatFormantTimeGrid, resampleForFormants } from '../src/js/local-inference/praat-analysis.js'
import { calculateSpectralTilt, computeMelSpectrogram } from '../src/js/local-inference/signal-analysis.js'
import { resolveModelUrls } from '../src/js/local-inference/model-urls.js'

const referenceCurves = Object.fromEntries(['pitch', 'f1', 'f2', 'f3', 'tilt'].map(feature => [feature, {
  f: [[0, 10000], [0.5, 0.5]],
  m: [[0, 10000], [0.5, 0.5]],
}]))

function referenceTfioMelFrame(samples, sampleRate, fftSize = 2048, bands = 128) {
  const magnitudes = new Float64Array(fftSize / 2 + 1)
  for (let bin = 0; bin <= fftSize / 2; bin += 1) {
    let real = 0
    let imag = 0
    for (let sample = 0; sample < fftSize; sample += 1) {
      const window = 0.5 - 0.5 * Math.cos(2 * Math.PI * sample / fftSize)
      const value = samples[sample] * window
      const angle = 2 * Math.PI * bin * sample / fftSize
      real += value * Math.cos(angle)
      imag -= value * Math.sin(angle)
    }
    magnitudes[bin] = Math.hypot(real, imag)
  }

  const hzToMel = hz => 2595 * Math.log10(1 + hz / 700)
  const upperHz = Math.min(8000, sampleRate / 2)
  const lowerMel = hzToMel(0)
  const upperMel = hzToMel(upperHz)
  const edges = Array.from({ length: bands + 2 }, (_, i) => lowerMel + (upperMel - lowerMel) * i / (bands + 1))
  return Array.from({ length: bands }, (_, band) => {
    const lower = edges[band]
    const center = edges[band + 1]
    const upper = edges[band + 2]
    let sum = 0
    for (let bin = 1; bin <= fftSize / 2; bin += 1) {
      const frequencyMel = hzToMel(bin * sampleRate / fftSize)
      const weight = Math.max(0, Math.min(
        (frequencyMel - lower) / (center - lower),
        (upper - frequencyMel) / (upper - center),
      ))
      sum += magnitudes[bin] * weight
    }
    return sum
  })
}

test('Praat formant frame grid uses the full default 50 ms analysis window', () => {
  const grid = praatFormantTimeGrid(5, 0.01)
  assert.equal(grid.count, 496)
  assert.ok(Math.abs(grid.first - 0.025) < 1e-12)
})

test('Praat formant frame grid reproduces sample-count rounding at two seconds', () => {
  const grid = praatFormantTimeGrid(2, 0.01)
  assert.equal(grid.count, 195)
  assert.ok(Math.abs(grid.first - 0.03) < 1e-12)
})

test('formant resampling matches the Praat backend at edges and interior samples', () => {
  const wav = readFileSync(fileURLToPath(new URL('../public/e2e-fixture.wav', import.meta.url)))
  let offset = 12
  let sampleRate = 0
  let dataOffset = 0
  let dataSize = 0
  while (offset + 8 <= wav.length) {
    const id = wav.toString('ascii', offset, offset + 4)
    const size = wav.readUInt32LE(offset + 4)
    if (id === 'fmt ') sampleRate = wav.readUInt32LE(offset + 12)
    if (id === 'data') { dataOffset = offset + 8; dataSize = size; break }
    offset += 8 + size + (size & 1)
  }
  const samples = new Float32Array(dataSize / 2)
  for (let index = 0; index < samples.length; index += 1) samples[index] = wav.readInt16LE(dataOffset + index * 2) / 32768
  const resampled = resampleForFormants(samples, sampleRate)
  for (const [index, expected] of [
    [0, 0.001951013171456865], [1, 0.011875351021920194], [2, 0.021907987388991215],
    [4, 0.04138138707748396], [100, -0.07658076568883009], [1000, 0.07447614874664132],
    [10000, -0.07657813037925242], [21999, -0.008315809253186621],
  ]) assert.ok(Math.abs(resampled[index] - expected) < 1e-7, `sample ${index}: expected ${expected}, got ${resampled[index]}`)
})

test('formants match the Python backend on the checked-in PCM fixture', () => {
  const wav = readFileSync(fileURLToPath(new URL('../public/e2e-fixture.wav', import.meta.url)))
  let offset = 12
  let sampleRate = 0
  let channels = 0
  let bitsPerSample = 0
  let dataOffset = 0
  let dataSize = 0
  while (offset + 8 <= wav.length) {
    const id = wav.toString('ascii', offset, offset + 4)
    const size = wav.readUInt32LE(offset + 4)
    if (id === 'fmt ') {
      channels = wav.readUInt16LE(offset + 10)
      sampleRate = wav.readUInt32LE(offset + 12)
      bitsPerSample = wav.readUInt16LE(offset + 22)
    }
    if (id === 'data') {
      dataOffset = offset + 8
      dataSize = size
      break
    }
    offset += 8 + size + (size & 1)
  }
  assert.equal(bitsPerSample, 16)
  const frameCount = dataSize / (bitsPerSample / 8 * channels)
  const samples = new Float32Array(frameCount)
  for (let frame = 0; frame < frameCount; frame += 1) {
    let mixed = 0
    for (let channel = 0; channel < channels; channel += 1) {
      mixed += wav.readInt16LE(dataOffset + (frame * channels + channel) * 2) / 32768
    }
    samples[frame] = mixed / channels
  }

  const reference = JSON.parse(readFileSync(fileURLToPath(new URL('./fixtures/praat-formant-reference.json', import.meta.url)), 'utf8'))
  assert.equal(sampleRate, reference.sample_rate)
  const rows = calculatePraatFrequencyArray(samples, sampleRate)
  assert.equal(rows.length, reference.frames)
  for (let frame = 0; frame < reference.frames; frame += 1) {
    for (let formant = 0; formant < 3; formant += 1) {
      assert.equal(rows[frame][formant + 1], reference.rows[frame][formant],
        `frame ${frame}, F${formant + 1}: expected ${reference.rows[frame][formant]}, got ${rows[frame][formant + 1]}`)
    }
  }
})

test('feature means use NumPy float32 pairwise accumulation semantics', () => {
  const values = [
    1506.8575439453125, 1139.6015625, 825.0381469726562, 670.5896606445312,
    916.1697998046875, 811.261474609375, 637.5274658203125, 676.9715576171875,
    564.64111328125, 223.94403076171875, Number.NaN,
  ]
  assert.equal(meanNumpyFloat32(values), 797.2601928710938)
})

// Golden Float32 output from TensorFlow 2.10.1 + TensorFlow-I/O 0.27.0 on the same PCM fixture.
test('Mel spectrogram matches the backend TensorFlow-I/O fixture across all frames', () => {
  const wav = readFileSync(fileURLToPath(new URL('../public/e2e-fixture.wav', import.meta.url)))
  const samples = Float32Array.from({length: 32000}, (_, index) => wav.readInt16LE(44 + index * 2) / 32768)
  const bytes = readFileSync(fileURLToPath(new URL('./fixtures/tfio-mel-e2e.f32', import.meta.url)))
  const expected = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))
  const actual = computeMelSpectrogram(samples, 16000)
  assert.equal(actual.length, 63)
  assert.equal(expected.length, actual.length * 128)
  let maxAbs = 0
  let absoluteSum = 0
  for (let frame = 0; frame < actual.length; frame += 1) {
    for (let band = 0; band < 128; band += 1) {
      const difference = Math.abs(Math.fround(actual[frame][band]) - expected[frame * 128 + band])
      maxAbs = Math.max(maxAbs, difference)
      absoluteSum += difference
    }
  }
  const mae = absoluteSum / expected.length
  assert.ok(mae <= 1e-5 && maxAbs <= 3e-4,
    `TFIO parity: MAE ${mae}, maximum absolute error ${maxAbs}`)
})

test('Mel spectrogram matches TensorFlow-I/O magnitude, padding, and HTK filterbank semantics', () => {
  const sampleRate = 16000
  const fftSize = 2048
  const samples = Float32Array.from({ length: fftSize }, (_, i) => 0.3 * Math.sin(2 * Math.PI * 1000 * i / sampleRate))
  const actual = computeMelSpectrogram(samples, sampleRate)
  const expected = referenceTfioMelFrame(samples, sampleRate)

  assert.equal(actual.length, Math.ceil(samples.length / 512), 'pad_end=true emits the final partial frames')
  assert.equal(actual[0].length, 128)
  for (let band = 0; band < expected.length; band += 1) {
    assert.ok(Math.abs(actual[0][band] - expected[band]) <= Math.max(1e-5, expected[band] * 1e-6),
      `mel band ${band}: expected ${expected[band]}, got ${actual[0][band]}`)
  }
})

test('buildLocalAnalysis returns all fields consumed by the results route without a server', async () => {
  const samples = new Float32Array(16000 * 2)
  for (let i = 0; i < samples.length; i += 1) samples[i] = 0.15 * Math.sin(2 * Math.PI * 180 * i / 16000)

  const result = await buildLocalAnalysis(samples, 16000, 2, { infer: async () => [], curves: referenceCurves })

  assert.deepEqual(Object.keys(result).sort(), ['audio_dur', 'freqArrays', 'inference_backend', 'ml', 'ml_enabled', 'ml_error', 'result', 'spec', 'spec_sr'])
  assert.equal(result.audio_dur, 2)
  assert.equal(result.spec_sr, 16000)
  assert.ok(result.spec.length > 0)
  assert.equal(result.spec[0].length, 128)
  assert.deepEqual(Object.keys(result.freqArrays).sort(), ['f1', 'f2', 'f3', 'pitch'])
  assert.ok(result.freqArrays.pitch instanceof Float32Array)
  assert.deepEqual(Object.keys(result.result), ['means', 'fem_prob'])
  assert.deepEqual(Object.keys(result.result.means).sort(), ['f1', 'f2', 'f3', 'pitch', 'tilt'])
  assert.equal(result.ml_error, null)
  assert.deepEqual(result.ml, [])
})

test('local statistics and spectrogram survive a model inference failure', async () => {
  const samples = new Float32Array(16000)
  for (let i = 0; i < samples.length; i += 1) samples[i] = 0.1 * Math.sin(2 * Math.PI * 160 * i / 16000)
  const result = await buildLocalAnalysis(samples, 16000, 1, {
    infer: async () => { throw new Error('model unavailable') },
    curves: referenceCurves,
  })
  assert.equal(result.ml_enabled, true)
  assert.equal(result.ml_error, 'model unavailable')
  assert.equal(result.ml.length, 0)
  assert.ok(result.spec.length > 0)
  assert.ok(result.freqArrays.pitch.length > 0)
})

test('buildLocalAnalysis keeps missing pitch values as NaN instead of converting them to zero', async () => {
  const samples = new Float32Array(16000)
  const result = await buildLocalAnalysis(samples, 16000, 1, { infer: async () => [], curves: referenceCurves })
  assert.ok(result.freqArrays.pitch.some(Number.isNaN))
})

test('spectral tilt supports longer recordings without spreading the full spectrum onto the call stack', () => {
  const length = 140000
  const samples = Float32Array.from({length}, (_, i) => 0.1 * Math.sin(2 * Math.PI * 180 * i / 16000))
  assert.ok(Number.isFinite(calculateSpectralTilt(samples)))
})

test('model URLs resolve from an origin without relying on the window global', () => {
  assert.equal(typeof window, 'undefined')
  assert.deepEqual(resolveModelUrls('/voice/', 'https://voice.hydev.org'), {
    vad: 'https://voice.hydev.org/voice/models/tfjs/keras_speech_music_noise_cnn/model.json',
    gender: 'https://voice.hydev.org/voice/models/tfjs/keras_male_female_cnn/model.json',
  })
})
