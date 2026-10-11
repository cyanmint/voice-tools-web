const FFT_SIZE = 512;
const WINDOW = 400;
const HOP = 160;
const FILTERS = 24;
const LOW_HZ = 100;
const HIGH_HZ = 8000;
const FRAME_WINDOW = 68;
const FRAME_STRIDE = 2;

const hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
const melToHz = (mel) => 700 * (10 ** (mel / 2595) - 1);

function melFilterBank(sampleRate) {
  const low = hzToMel(LOW_HZ);
  const high = hzToMel(Math.min(HIGH_HZ, sampleRate / 2));
  const frequencies = Array.from({ length: FILTERS + 2 }, (_, i) => melToHz(low + i * (high - low) / (FILTERS + 1)));
  return Array.from({ length: FILTERS }, (_, filter) => {
    const [left, center, right] = frequencies.slice(filter, filter + 3);
    const height = 2 / (right - left);
    const weights = new Float64Array(FFT_SIZE / 2 + 1);
    const leftStart = Math.floor(left * FFT_SIZE / sampleRate) + 1;
    const centerEnd = Math.floor(center * FFT_SIZE / sampleRate);
    for (let bin = leftStart; bin <= centerEnd && bin < weights.length; bin += 1) {
      weights[bin] = height / (center - left) * (bin * sampleRate / FFT_SIZE - left);
    }
    const rightStart = Math.floor(center * FFT_SIZE / sampleRate) + 1;
    const rightEndExclusive = Math.min(Math.floor(right * FFT_SIZE / sampleRate) + 1, FFT_SIZE);
    for (let bin = rightStart; bin < rightEndExclusive - 1 && bin < weights.length; bin += 1) {
      weights[bin] = height / (right - center) * (right - bin * sampleRate / FFT_SIZE);
    }
    return weights;
  });
}

function powerSpectrum(frame) {
  const real = new Float64Array(FFT_SIZE);
  const imag = new Float64Array(FFT_SIZE);
  for (let i = 0; i < WINDOW; i += 1) {
    const hann = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / WINDOW);
    real[i] = frame[i] * hann;
  }
  for (let i = 1, j = 0; i < FFT_SIZE; i += 1) {
    let bit = FFT_SIZE >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }
  for (let size = 2; size <= FFT_SIZE; size <<= 1) {
    const angle = -2 * Math.PI / size;
    const half = size >> 1;
    for (let base = 0; base < FFT_SIZE; base += size) {
      for (let offset = 0; offset < half; offset += 1) {
        const wr = Math.cos(angle * offset);
        const wi = Math.sin(angle * offset);
        const even = base + offset;
        const odd = even + half;
        const tr = wr * real[odd] - wi * imag[odd];
        const ti = wr * imag[odd] + wi * real[odd];
        real[odd] = real[even] - tr;
        imag[odd] = imag[even] - ti;
        real[even] += tr;
        imag[even] += ti;
      }
    }
  }
  return Float64Array.from({ length: FFT_SIZE / 2 + 1 }, (_, i) => real[i] ** 2 + imag[i] ** 2);
}

/** Extract INA-style 24-band log-Mel features from mono float PCM. */
export function extractMelFeatures(audio, { sampleRate = 16000, preEmphasis = 0.97 } = {}) {
  if (!(audio instanceof Float32Array) && !Array.isArray(audio)) throw new TypeError('audio must be Float32Array or number[]');
  if (sampleRate !== 16000) throw new RangeError('INA models require 16 kHz audio; resampling is not implemented in this experiment');
  if (!audio.length || !Number.isFinite(preEmphasis) || preEmphasis < 0) throw new RangeError('audio must be non-empty and preEmphasis non-negative');
  const emphasized = new Float64Array(audio.length);
  emphasized[0] = audio[0];
  for (let i = 1; i < audio.length; i += 1) emphasized[i] = audio[i] - audio[i - 1] * preEmphasis;
  const filters = melFilterBank(sampleRate);
  const rows = [];
  const logEnergy = [];
  for (let start = 0; start + WINDOW <= emphasized.length; start += HOP) {
    const frame = emphasized.subarray(start, start + WINDOW);
    let energy = 0;
    for (const sample of frame) energy += sample * sample;
    logEnergy.push(Math.log(energy));
    const power = powerSpectrum(frame);
    rows.push(filters.map((weights) => {
      let sum = 0;
      for (let bin = 0; bin < power.length; bin += 1) sum += power[bin] * weights[bin];
      return Math.log(sum);
    }));
  }
  if (rows.length < FRAME_WINDOW) throw new RangeError(`audio is too short for INA's 68-frame CNN window (${rows.length} frames)`);
  return { mel: rows, logEnergy, frameStepSeconds: HOP / sampleRate };
}

/** Reproduce INA's 68-frame windows, per-window normalization, and edge padding. */
export function makeFeatureWindows(mel, { featureCount = 24 } = {}) {
  if (!Array.isArray(mel) || mel.length < FRAME_WINDOW || !Number.isInteger(featureCount) || featureCount < 1) {
    throw new RangeError('mel must contain at least 68 frames and featureCount must be positive');
  }
  if (mel.some((row) => row.length < featureCount)) throw new RangeError(`mel rows must contain at least ${featureCount} bands`);
  const windows = [];
  const finite = [];
  for (let start = 0; start + FRAME_WINDOW <= mel.length; start += FRAME_STRIDE) {
    const flat = [];
    for (let t = start; t < start + FRAME_WINDOW; t += 1) for (let f = 0; f < featureCount; f += 1) flat.push(mel[t][f]);
    const mean = flat.reduce((sum, value) => sum + value, 0) / flat.length;
    const variance = flat.reduce((sum, value) => sum + (value - mean) ** 2, 0) / flat.length;
    const std = Math.sqrt(variance);
    const normalized = flat.map((value) => (value - mean) / std);
    const isFinite = normalized.every(Number.isFinite);
    finite.push(isFinite);
    windows.push(Array.from({ length: FRAME_WINDOW }, (_, t) =>
      Array.from({ length: featureCount }, (_, f) => isFinite ? normalized[t * featureCount + f] : 0)));
  }
  const leftFill = Math.floor(FRAME_WINDOW / (2 * FRAME_STRIDE));
  const rightFill = leftFill - 1 + (mel.length % 2);
  const first = windows[0];
  const last = windows.at(-1);
  const paddedWindows = [
    ...Array.from({ length: leftFill }, () => first),
    ...windows,
    ...Array.from({ length: rightFill }, () => last),
  ];
  const paddedFinite = [
    ...Array.from({ length: leftFill }, () => finite[0]),
    ...finite,
    ...Array.from({ length: rightFill }, () => finite.at(-1)),
  ];
  return { windows: paddedWindows, finite: paddedFinite };
}
