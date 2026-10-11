const clamp = (value, min, max) => Math.max(min, Math.min(max, value))

const FFT_TWIDDLES = new Map()

function twiddlesFor(size) {
  let twiddles = FFT_TWIDDLES.get(size)
  if (!twiddles) {
    twiddles = new Map()
    for (let stage = 2; stage <= size; stage <<= 1) {
      const half = stage >> 1
      const real = new Float64Array(half)
      const imag = new Float64Array(half)
      for (let offset = 0; offset < half; offset += 1) {
        const angle = -2 * Math.PI * offset / stage
        real[offset] = Math.cos(angle)
        imag[offset] = Math.sin(angle)
      }
      twiddles.set(stage, { real, imag })
    }
    FFT_TWIDDLES.set(size, twiddles)
  }
  return twiddles
}

function fftInPlace(real, imag, inverse = false) {
  const length = real.length
  const twiddles = twiddlesFor(length)
  for (let i = 1, j = 0; i < length; i += 1) {
    let bit = length >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      ;[real[i], real[j]] = [real[j], real[i]]
      ;[imag[i], imag[j]] = [imag[j], imag[i]]
    }
  }
  for (let size = 2; size <= length; size <<= 1) {
    const half = size >> 1
    const stageTwiddles = twiddles.get(size)
    for (let base = 0; base < length; base += size) {
      for (let offset = 0; offset < half; offset += 1) {
        const wr = stageTwiddles.real[offset]
        const wi = inverse ? -stageTwiddles.imag[offset] : stageTwiddles.imag[offset]
        const even = base + offset
        const odd = even + half
        const tr = wr * real[odd] - wi * imag[odd]
        const ti = wr * imag[odd] + wi * real[odd]
        real[odd] = real[even] - tr
        imag[odd] = imag[even] - ti
        real[even] += tr
        imag[even] += ti
      }
    }
  }
  if (inverse) for (let i = 0; i < length; i += 1) { real[i] /= length; imag[i] /= length }
}

function nextPowerOfTwo(value) {
  let result = 1
  while (result < value) result <<= 1
  return result
}

function gridForDuration(duration, width, step) {
  const count = Math.max(0, Math.floor((duration - width) / step + 1e-10) + 1)
  return { count, first: count ? (duration - (count - 1) * step) / 2 : 0 }
}

function sampleLinear(samples, position) {
  if (position < 0 || position > samples.length - 1) return 0
  const lower = Math.floor(position)
  const ratio = position - lower
  return samples[lower] * (1 - ratio) + (samples[lower + 1] ?? 0) * ratio
}

function autocorrelation(samples) {
  const length = nextPowerOfTwo(samples.length * 1.5)
  const real = new Float64Array(length)
  const imag = new Float64Array(length)
  for (let i = 0; i < samples.length; i += 1) real[i] = samples[i]
  fftInPlace(real, imag)
  for (let i = 0; i < length; i += 1) real[i] = real[i] * real[i] + imag[i] * imag[i]
  imag.fill(0)
  fftInPlace(real, imag, true)
  return real
}

function peakParabola(values, index) {
  const left = values[index - 1]
  const center = values[index]
  const right = values[index + 1]
  const denominator = left - 2 * center + right
  const offset = Math.abs(denominator) > 1e-14 ? clamp(0.5 * (left - right) / denominator, -0.5, 0.5) : 0
  return { lag: index + offset, strength: center - 0.25 * (left - right) * offset }
}

function pitchFrameCandidates(samples, sampleRate, centre, globalPeak, config, windowAc) {
  const windowSamples = config.pitchWindowSamples
  const half = Math.floor(windowSamples / 2)
  const start = Math.round(centre * sampleRate - half)
  const periodSpan = Math.floor(sampleRate / config.pitchFloor)
  let localMean = 0
  let meanCount = 0
  for (let i = Math.max(0, start - periodSpan); i < Math.min(samples.length, start + windowSamples + periodSpan); i += 1) {
    localMean += samples[i]
    meanCount += 1
  }
  if (meanCount) localMean /= meanCount
  const raw = new Float64Array(config.pitchWindowSamples)
  for (let i = 0; i < raw.length; i += 1) {
    const sample = sampleLinear(samples, start + i) - localMean
    raw[i] = sample * config.pitchWindow[i]
  }
  let peak = 0
  for (let i = Math.max(0, start); i < Math.min(samples.length, start + windowSamples); i += 1) peak = Math.max(peak, Math.abs(samples[i] - localMean))
  if (globalPeak === 0 || peak < globalPeak * config.silenceThreshold) return [{ frequency: 0, strength: 0, score: config.voicingThreshold }]

  const ac = autocorrelation(raw)
  const ac0 = ac[0]
  if (!(ac0 > 1e-20)) return [{ frequency: 0, strength: 0, score: config.voicingThreshold }]
  const minLag = Math.max(2, Math.floor(sampleRate / config.pitchCeiling))
  const maxLag = Math.min(Math.floor(raw.length / 3) + 2, raw.length - 1)
  const correlations = new Float64Array(maxLag + 2)
  for (let lag = minLag - 1; lag <= maxLag + 1; lag += 1) {
    const correction = windowAc[lag] / Math.max(windowAc[0], 1e-20)
    correlations[lag] = correction > 1e-8 ? ac[lag] / ac0 / correction : 0
  }
  const peaks = []
  for (let lag = minLag; lag <= maxLag; lag += 1) {
    if (correlations[lag] >= correlations[lag - 1] && correlations[lag] > correlations[lag + 1]) {
      const peakAtLag = peakParabola(correlations, lag)
      if (peakAtLag.strength > 0) {
        const frequency = sampleRate / peakAtLag.lag
        if (frequency >= config.pitchFloor && frequency <= config.pitchCeiling) {
          peaks.push({ frequency, strength: peakAtLag.strength, score: peakAtLag.strength - config.octaveCost * Math.log2(config.pitchCeiling / frequency) })
        }
      }
    }
  }
  peaks.sort((a, b) => b.strength - a.strength)
  peaks.length = Math.min(peaks.length, config.maxCandidates - 1)
  return [{ frequency: 0, strength: 0, score: config.voicingThreshold }, ...peaks]
}

function pitchTrack(samples, sampleRate, timeStep) {
  const config = {
    pitchFloor: 75,
    pitchCeiling: Math.min(600, sampleRate / 2),
    pitchWindowSamples: Math.max(8, Math.floor((3 / 75) * sampleRate / 2 - 1) * 2),
    maxCandidates: 15,
    silenceThreshold: 0.03,
    voicingThreshold: 0.45,
    octaveCost: 0.01,
    octaveJumpCost: 0.35 * 0.01 / timeStep,
    voicedUnvoicedCost: 0.14 * 0.01 / timeStep,
  }
  const duration = samples.length / sampleRate
  const effectiveWindowSeconds = 3 / config.pitchFloor
  const grid = gridForDuration(duration, effectiveWindowSeconds, timeStep)
  let average = 0
  let globalPeak = 0
  for (const value of samples) average += value
  average /= Math.max(samples.length, 1)
  for (const value of samples) globalPeak = Math.max(globalPeak, Math.abs(value - average))
  const pitchWindow = Float64Array.from({ length: config.pitchWindowSamples }, (_, i) =>
    0.5 - 0.5 * Math.cos(2 * Math.PI * (i + 1) / (config.pitchWindowSamples + 1)))
  config.pitchWindow = pitchWindow
  const windowAc = autocorrelation(pitchWindow)
  const frames = Array.from({ length: grid.count }, (_, i) =>
    pitchFrameCandidates(samples, sampleRate, grid.first + i * timeStep, globalPeak, config, windowAc))
  if (!frames.length) return { values: [], times: [], candidates: frames }

  const costs = frames.map(frame => new Float64Array(frame.length))
  const back = frames.map(frame => new Int32Array(frame.length))
  for (let candidate = 0; candidate < frames[0].length; candidate += 1) costs[0][candidate] = -frames[0][candidate].score
  for (let frame = 1; frame < frames.length; frame += 1) {
    for (let current = 0; current < frames[frame].length; current += 1) {
      let bestCost = Infinity
      let bestPrevious = 0
      const currentCandidate = frames[frame][current]
      for (let previous = 0; previous < frames[frame - 1].length; previous += 1) {
        const previousCandidate = frames[frame - 1][previous]
        let transition = 0
        if (currentCandidate.frequency === 0 || previousCandidate.frequency === 0) {
          if (currentCandidate.frequency !== previousCandidate.frequency) transition = config.voicedUnvoicedCost
        } else {
          transition = config.octaveJumpCost * Math.abs(Math.log2(currentCandidate.frequency / previousCandidate.frequency))
        }
        const cost = costs[frame - 1][previous] + transition - currentCandidate.score
        if (cost < bestCost) { bestCost = cost; bestPrevious = previous }
      }
      costs[frame][current] = bestCost
      back[frame][current] = bestPrevious
    }
  }
  let state = 0
  const last = costs.at(-1)
  for (let i = 1; i < last.length; i += 1) if (last[i] < last[state]) state = i
  const selected = new Int32Array(frames.length)
  selected[selected.length - 1] = state
  for (let frame = selected.length - 1; frame > 0; frame -= 1) selected[frame - 1] = back[frame][selected[frame]]
  return {
    values: Array.from(selected, (candidate, i) => frames[i][candidate].frequency || null),
    times: Array.from({ length: grid.count }, (_, i) => grid.first + i * timeStep),
    candidates: frames,
  }
}

function sinc(x) {
  return Math.abs(x) < 1e-12 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x)
}

function interpolateSinc(samples, index, precision) {
  if (index < 1) return samples[0]
  if (index > samples.length) return samples[samples.length - 1]
  const leftMid = Math.floor(index)
  const rightMid = leftMid + 1
  if (index === leftMid) return samples[leftMid - 1]

  const depth = Math.min(precision, rightMid - 1, samples.length - leftMid)
  if (depth <= 0) return samples[Math.floor(index + 0.5) - 1]
  if (depth === 1) {
    const fraction = index - leftMid
    return samples[leftMid - 1] + fraction * (samples[rightMid - 1] - samples[leftMid - 1])
  }
  if (depth === 2) {
    const leftValue = samples[leftMid - 1]
    const rightValue = samples[rightMid - 1]
    const leftSlope = 0.5 * (rightValue - samples[leftMid - 2])
    const rightSlope = 0.5 * (samples[rightMid] - leftValue)
    const fractionLeft = index - leftMid
    const fractionRight = rightMid - index
    return leftValue * fractionRight + rightValue * fractionLeft - fractionLeft * fractionRight * (
      0.5 * (rightSlope - leftSlope) + (fractionLeft - 0.5) * (leftSlope + rightSlope - 2 * (rightValue - leftValue))
    )
  }

  const left = rightMid - depth
  const right = leftMid + depth
  let result = 0
  const leftRadius = index - left + 1
  for (let sourceIndex = leftMid; sourceIndex >= left; sourceIndex -= 1) {
    const distance = index - sourceIndex
    const window = 0.5 * (1 + Math.cos(Math.PI * distance / leftRadius))
    result += samples[sourceIndex - 1] * sinc(distance) * window
  }
  const rightRadius = right - index + 1
  for (let sourceIndex = rightMid; sourceIndex <= right; sourceIndex += 1) {
    const distance = sourceIndex - index
    const window = 0.5 * (1 + Math.cos(Math.PI * distance / rightRadius))
    result += samples[sourceIndex - 1] * sinc(distance) * window
  }
  return result
}

export function resampleForFormants(samples, inputRate, outputRate = 11000) {
  if (inputRate === outputRate) return Float64Array.from(samples)
  const outputLength = Math.round(samples.length * outputRate / inputRate)
  let filtered = samples

  if (outputRate < inputRate) {
    const antiTurnaround = 1000
    const fftSize = nextPowerOfTwo(samples.length + antiTurnaround * 2)
    const real = new Float64Array(fftSize)
    const imag = new Float64Array(fftSize)
    real.set(samples, antiTurnaround)
    fftInPlace(real, imag)

    const cutoff = Math.floor(outputRate / inputRate * fftSize)
    for (let bin = 1; bin < fftSize / 2; bin += 1) {
      const realIndex = 2 * bin
      const imaginaryIndex = realIndex + 1
      if (realIndex >= cutoff - 1) {
        real[bin] = 0
        imag[bin] = 0
        real[fftSize - bin] = 0
        imag[fftSize - bin] = 0
      } else if (imaginaryIndex >= cutoff - 1) {
        imag[bin] = 0
        imag[fftSize - bin] = 0
      }
    }
    real[fftSize / 2] = 0
    imag[fftSize / 2] = 0
    fftInPlace(real, imag, true)
    filtered = real.subarray(antiTurnaround, antiTurnaround + samples.length)
  }

  const duration = samples.length / inputRate
  const inputFirstSampleTime = 0.5 / inputRate
  const outputFirstSampleTime = 0.5 * (duration - (outputLength - 1) / outputRate)
  const output = new Float64Array(outputLength)
  for (let i = 0; i < outputLength; i += 1) {
    const time = outputFirstSampleTime + i / outputRate
    const sourceIndex = (time - inputFirstSampleTime) * inputRate + 1
    output[i] = interpolateSinc(filtered, sourceIndex, 50)
  }
  return output
}

function normalizeLpcCoefficients(coefficients) {
  const normalized = Float64Array.from(coefficients)
  for (let index = 1; index < normalized.length; index += 1) normalized[index] = -normalized[index]
  return Array.from(normalized)
}

function burgLpc(frame, order) {
  const coefficients = new Float64Array(order + 1)
  coefficients[0] = 1
  if (frame.length < 2) return null

  let forward = Float64Array.from(frame.subarray(0, frame.length - 1))
  let backward = Float64Array.from(frame.subarray(1))
  for (let m = 1; m <= order; m += 1) {
    let numerator = 0
    let denominator = 0
    for (let n = 0; n < forward.length; n += 1) {
      numerator += forward[n] * backward[n]
      denominator += forward[n] * forward[n] + backward[n] * backward[n]
    }
    if (denominator < 1e-24) return null
    const reflection = clamp(2 * numerator / denominator, -0.999, 0.999)
    const previous = coefficients.slice()
    for (let j = 1; j < m; j += 1) coefficients[j] = previous[j] - reflection * previous[m - j]
    coefficients[m] = reflection

    if (m < order) {
      const nextForward = new Float64Array(forward.length - 1)
      const nextBackward = new Float64Array(backward.length - 1)
      for (let n = 0; n < nextForward.length; n += 1) {
        nextForward[n] = forward[n] - reflection * backward[n]
        nextBackward[n] = backward[n + 1] - reflection * forward[n + 1]
      }
      forward = nextForward
      backward = nextBackward
    }
  }
  return coefficients
}

function cAdd(a, b) { return [a[0] + b[0], a[1] + b[1]] }
function cMul(a, b) { return [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]] }
function cSub(a, b) { return [a[0] - b[0], a[1] - b[1]] }
function cDiv(a, b) {
  const denominator = b[0] * b[0] + b[1] * b[1]
  return denominator < 1e-30 ? [0, 0] : [(a[0] * b[0] + a[1] * b[1]) / denominator, (a[1] * b[0] - a[0] * b[1]) / denominator]
}

function evaluatePolynomial(coefficients, [x, y]) {
  let value = [coefficients[0], 0]
  let derivative = [0, 0]
  for (let index = 1; index < coefficients.length; index += 1) {
    derivative = cAdd(cMul(derivative, [x, y]), value)
    value = cAdd(cMul(value, [x, y]), [coefficients[index], 0])
  }
  return { value, derivative }
}

function polynomialRoots(coefficients) {
  const degree = coefficients.length - 1
  const radius = 0.98
  let roots = Array.from({ length: degree }, (_, index) => {
    const angle = 2 * Math.PI * (index + 0.25) / degree
    return [radius * Math.cos(angle), radius * Math.sin(angle)]
  })
  for (let iteration = 0; iteration < 800; iteration += 1) {
    let largestDelta = 0
    const deltas = roots.map((root, index) => {
      const { value, derivative } = evaluatePolynomial(coefficients, root)
      const newton = cDiv(value, derivative)
      let sum = [0, 0]
      for (let other = 0; other < degree; other += 1) {
        if (other !== index) sum = cAdd(sum, cDiv([1, 0], cSub(root, roots[other])))
      }
      const correction = cDiv(newton, cSub([1, 0], cMul(newton, sum)))
      largestDelta = Math.max(largestDelta, Math.hypot(correction[0], correction[1]))
      return correction
    })
    roots = roots.map((root, index) => cSub(root, deltas[index]))
    if (largestDelta < 1e-12) break
  }
  return roots
}

function formantsInFrame(samples, sampleRate, centre) {
  const effectiveLength = 0.025
  const physicalLength = 2 * effectiveLength
  const windowSize = Math.max(12, Math.floor(physicalLength * sampleRate))
  const halfWindow = Math.floor(windowSize / 2)
  const start = Math.floor(centre * sampleRate - 0.5) + 1 - halfWindow
  const frame = new Float64Array(windowSize)
  const edge = Math.exp(-12)
  const preEmphasis = Math.exp(-2 * Math.PI * 50 / sampleRate)
  for (let i = 0; i < windowSize; i += 1) {
    const index = start + i
    const current = index >= 0 && index < samples.length ? samples[index] : 0
    const previousIndex = index - 1
    const previous = previousIndex >= 0 && previousIndex < samples.length ? samples[previousIndex] : 0
    const emphasized = current - preEmphasis * previous
    const offset = (i + 1) - 0.5 * (windowSize + 1)
    const gaussian = (Math.exp(-48 * offset * offset / ((windowSize + 1) * (windowSize + 1))) - edge) / (1 - edge)
    frame[i] = emphasized * gaussian
  }
  const coefficients = burgLpc(frame, 10)
  if (!coefficients) return []
  const roots = polynomialRoots(normalizeLpcCoefficients(coefficients))
  const nyquist = sampleRate / 2
  return roots.filter(([, imaginary]) => imaginary > 1e-8).map(([real, imaginary]) => {
    const magnitude = Math.hypot(real, imaginary)
    const frequency = Math.abs(Math.atan2(imaginary, real)) * nyquist / Math.PI
    const bandwidth = -Math.log(Math.max(magnitude, 1e-12)) * nyquist / Math.PI
    return { frequency, bandwidth }
  }).filter(({ frequency }) => frequency >= 50 && frequency <= nyquist - 50)
    .sort((a, b) => a.frequency - b.frequency)
    .map(({ frequency }) => frequency)
}

function formantAtTime(formantFrames, frameTimes, formantNumber, time, timeStep) {
  const coordinate = (time - frameTimes[0]) / timeStep
  if (coordinate < -0.5 || coordinate > frameTimes.length - 0.5) return null
  const left = clamp(Math.floor(coordinate), 0, frameTimes.length - 1)
  const right = Math.min(frameTimes.length - 1, left + 1)
  const leftValue = formantFrames[left][formantNumber]
  const rightValue = formantFrames[right][formantNumber]
  if (left === right || leftValue == null || rightValue == null) {
    const nearest = clamp(Math.round(coordinate), 0, frameTimes.length - 1)
    return formantFrames[nearest][formantNumber] ?? null
  }
  const ratio = coordinate - left
  return leftValue + ratio * (rightValue - leftValue)
}

export function debugPraatFormantDetails(samples, sampleRate, time) {
  const resampled = resampleForFormants(samples, sampleRate)
  const effectiveLength = 0.025
  const windowSize = Math.max(12, Math.floor(2 * effectiveLength * 11000))
  const halfWindow = Math.floor(windowSize / 2)
  const start = Math.floor(time * 11000 - 0.5) + 1 - halfWindow
  const frame = new Float64Array(windowSize)
  const edge = Math.exp(-12)
  const emphasis = Math.exp(-2 * Math.PI * 50 / 11000)
  for (let i = 0; i < windowSize; i += 1) {
    const index = start + i
    const current = index >= 0 && index < resampled.length ? resampled[index] : 0
    const previous = index > 0 && index - 1 < resampled.length ? resampled[index - 1] : 0
    const offset = i - (windowSize - 1) / 2
    frame[i] = (current - emphasis * previous) * (Math.exp(-48 * offset * offset / ((windowSize + 1) ** 2)) - edge) / (1 - edge)
  }
  const coefficients = burgLpc(frame, 10)
  const values = Array.from(coefficients ?? [])
  return values
}

export function debugPraatFormants(samples, sampleRate, time) {
  const resampled = resampleForFormants(samples, sampleRate)
  return formantsInFrame(resampled, 11000, time)
}

export function praatFormantTimeGrid(duration, timeStep) {
  const sampleCount = Math.round(duration * 11000)
  const physicalDuration = sampleCount * (1 / 11000)
  const count = Math.max(0, Math.floor((physicalDuration - 0.05) / timeStep) + 1)
  return { count, first: count ? (physicalDuration - (count - 1) * timeStep) / 2 : 0 }
}

export function calculatePraatFrequencyArray(samples, sampleRate, timeStep = 0.01) {
  if (!(samples instanceof Float32Array) || !samples.length) throw new TypeError('samples must be a non-empty Float32Array')
  if (!Number.isFinite(sampleRate) || sampleRate < 11000) throw new RangeError('Praat formant analysis requires at least 11 kHz sample rate')
  if (!Number.isFinite(timeStep) || timeStep <= 0) throw new RangeError('timeStep must be positive')

  const duration = samples.length / sampleRate
  const pitch = pitchTrack(samples, sampleRate, timeStep)
  const resampled = resampleForFormants(samples, sampleRate)
  const resampledRate = 11000
  const formantGrid = praatFormantTimeGrid(duration, timeStep)
  const formantTimes = Array.from({ length: formantGrid.count }, (_, i) => formantGrid.first + i * timeStep)
  const formantFrames = formantTimes.map(time => formantsInFrame(resampled, resampledRate, time))
  return pitch.values.map((pitchValue, index) => {
    if (pitchValue == null) return [null, null, null, null]
    const queryTime = index * timeStep
    return [pitchValue,
      formantAtTime(formantFrames, formantTimes, 0, queryTime, timeStep),
      formantAtTime(formantFrames, formantTimes, 1, queryTime, timeStep),
      formantAtTime(formantFrames, formantTimes, 2, queryTime, timeStep),
    ].map(value => value == null || !Number.isFinite(value) ? null : Math.fround(value))
  })
}

export function calculatePraatSpectralTilt(samples) {
  if (!(samples instanceof Float32Array) || !samples.length) return null
  const size = nextPowerOfTwo(samples.length)
  const real = new Float64Array(size)
  const imag = new Float64Array(size)
  for (let i = 0; i < samples.length; i += 1) real[i] = samples[i]
  fftInPlace(real, imag)
  const bins = Math.floor(size / 2)
  const db = new Float64Array(bins)
  let maximum = -Infinity
  let minimum = Infinity
  for (let bin = 0; bin < bins; bin += 1) {
    const magnitude = Math.hypot(real[bin], imag[bin])
    if (!(magnitude > 0)) return null
    const value = 20 * Math.log10(magnitude / 0.0002)
    db[bin] = value
    maximum = Math.max(maximum, value)
    minimum = Math.min(minimum, value)
  }
  const range = maximum - minimum
  if (!(range > 0) || bins < 2) return null
  let sumX = 0
  let sumY = 0
  let sumXX = 0
  let sumXY = 0
  const factor = (bins - 1) / range
  for (let bin = 0; bin < bins; bin += 1) {
    const y = (db[bin] + Math.abs(minimum)) * factor
    sumX += bin
    sumY += y
    sumXX += bin * bin
    sumXY += bin * y
  }
  return (sumXY - sumX * sumY / bins) / (sumXX - sumX * sumX / bins)
}
