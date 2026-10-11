const clamp = (x, low, high) => Math.max(low, Math.min(high, x));

export function fft(real, imag) {
  const n = real.length;
  if (n !== imag.length || (n & (n - 1))) throw new RangeError('FFT arrays must have the same power-of-two length');
  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const theta = -2 * Math.PI / size;
    for (let base = 0; base < n; base += size) {
      for (let j = 0; j < half; j += 1) {
        const angle = theta * j;
        const wr = Math.cos(angle);
        const wi = Math.sin(angle);
        const even = base + j;
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
}

function hzToMel(hz) { return 2595 * Math.log10(1 + hz / 700); }

/**
 * TensorFlow-I/O-compatible magnitude Mel spectrogram.
 * tfio.audio.spectrogram uses a periodic Hann STFT with pad_end=true and
 * magnitude (not power); tfio.audio.melscale uses TF's HTK Mel-domain triangles.
 */
export function computeMelSpectrogram(samples, sampleRate, { fftSize = 2048, hopSize = 512, bands = 128 } = {}) {
  if (!(samples instanceof Float32Array) || !samples.length) throw new TypeError('samples must be non-empty Float32Array');
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) throw new RangeError('sampleRate must be positive');
  if (!Number.isInteger(fftSize) || fftSize < 2 || (fftSize & (fftSize - 1))) throw new RangeError('fftSize must be a power of two');
  if (!Number.isInteger(hopSize) || hopSize < 1) throw new RangeError('hopSize must be positive');
  if (!Number.isInteger(bands) || bands < 1) throw new RangeError('bands must be positive');

  const maxHz = Math.min(8000, sampleRate / 2);
  const melEdges = Array.from({ length: bands + 2 }, (_, i) =>
    hzToMel(0) + (hzToMel(maxHz) - hzToMel(0)) * i / (bands + 1));
  const filters = Array.from({ length: bands }, (_, band) => {
    const lower = melEdges[band];
    const center = melEdges[band + 1];
    const upper = melEdges[band + 2];
    const bins = [];
    const weights = [];
    for (let bin = 1; bin <= fftSize / 2; bin += 1) {
      const frequencyMel = hzToMel(bin * sampleRate / fftSize);
      const weight = Math.max(0, Math.min(
        (frequencyMel - lower) / (center - lower),
        (upper - frequencyMel) / (upper - center),
      ));
      if (weight > 0) {
        bins.push(bin);
        weights.push(weight);
      }
    }
    return { bins, weights };
  });

  const frameCount = Math.ceil(samples.length / hopSize);
  const rows = new Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    const start = frame * hopSize;
    const real = new Float64Array(fftSize);
    const imag = new Float64Array(fftSize);
    for (let i = 0; i < fftSize; i += 1) {
      const sampleIndex = start + i;
      const sample = sampleIndex < samples.length ? samples[sampleIndex] : 0;
      real[i] = sample * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / fftSize));
    }
    fft(real, imag);
    const magnitudes = new Float64Array(fftSize / 2 + 1);
    for (let bin = 1; bin <= fftSize / 2; bin += 1) magnitudes[bin] = Math.hypot(real[bin], imag[bin]);
    rows[frame] = filters.map(({ bins, weights }) => {
      let sum = 0;
      for (let i = 0; i < bins.length; i += 1) sum += weights[i] * magnitudes[bins[i]];
      return sum;
    });
  }
  return rows;
}

function frameSamples(samples, center, size) {
  const out = new Float64Array(size);
  const first = Math.round(center - size / 2);
  for (let i = 0; i < size; i += 1) {
    const source = first + i;
    if (source >= 0 && source < samples.length) out[i] = samples[source];
  }
  return out;
}

export function estimatePitch(samples, sampleRate, time = 0) {
  const size = Math.max(256, Math.round(sampleRate * 0.04));
  const frame = frameSamples(samples, time * sampleRate, size);
  let mean = 0;
  for (const x of frame) mean += x;
  mean /= size;
  for (let i = 0; i < size; i += 1) frame[i] = (frame[i] - mean) * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (size - 1)));
  let energy = 0;
  for (const x of frame) energy += x * x;
  if (energy / size < 1e-7) return null;
  const minLag = Math.max(2, Math.floor(sampleRate / 600));
  const maxLag = Math.min(size - 2, Math.ceil(sampleRate / 75));
  let bestLag = -1;
  let best = 0;
  for (let lag = minLag; lag <= maxLag; lag += 1) {
    let cross = 0;
    let left = 0;
    let right = 0;
    for (let i = 0; i < size - lag; i += 1) {
      cross += frame[i] * frame[i + lag];
      left += frame[i] ** 2;
      right += frame[i + lag] ** 2;
    }
    const score = cross / Math.sqrt(Math.max(left * right, 1e-20));
    if (score > best) { best = score; bestLag = lag; }
  }
  if (bestLag < 0 || best < 0.28) return null;
  let lag = bestLag;
  if (bestLag > minLag && bestLag < maxLag) {
    let ym = 0; let y0 = 0; let yp = 0;
    for (let i = 0; i < size - bestLag + 1; i += 1) ym += frame[i] * frame[i + bestLag - 1];
    for (let i = 0; i < size - bestLag; i += 1) y0 += frame[i] * frame[i + bestLag];
    for (let i = 0; i < size - bestLag - 1; i += 1) yp += frame[i] * frame[i + bestLag + 1];
    const denominator = ym - 2 * y0 + yp;
    if (Math.abs(denominator) > 1e-12) lag += clamp(0.5 * (ym - yp) / denominator, -0.5, 0.5);
  }
  return sampleRate / lag;
}

function levinsonDurbin(signal, order) {
  const r = new Float64Array(order + 1);
  for (let lag = 0; lag <= order; lag += 1) for (let i = lag; i < signal.length; i += 1) r[lag] += signal[i] * signal[i - lag];
  if (r[0] < 1e-10) return null;
  const a = new Float64Array(order + 1);
  a[0] = 1;
  let error = r[0];
  for (let i = 1; i <= order; i += 1) {
    let residual = r[i];
    for (let j = 1; j < i; j += 1) residual += a[j] * r[i - j];
    const reflection = clamp(-residual / Math.max(error, 1e-20), -0.98, 0.98);
    const previous = a.slice();
    for (let j = 1; j < i; j += 1) a[j] = previous[j] + reflection * previous[i - j];
    a[i] = reflection;
    error *= 1 - reflection * reflection;
  }
  return a;
}

const cmul = ([a, b], [c, d]) => [a * c - b * d, a * d + b * c];
const csub = ([a, b], [c, d]) => [a - c, b - d];
const cdiv = ([a, b], [c, d]) => { const den = c * c + d * d || 1e-24; return [(a * c + b * d) / den, (b * c - a * d) / den]; };

function polynomialRoots(coefficients) {
  const degree = coefficients.length - 1;
  const roots = Array.from({ length: degree }, (_, i) => {
    const angle = 2 * Math.PI * (i + 0.23) / degree;
    return [0.85 * Math.cos(angle), 0.85 * Math.sin(angle)];
  });
  for (let iteration = 0; iteration < 120; iteration += 1) {
    let largest = 0;
    for (let i = 0; i < degree; i += 1) {
      let value = [coefficients[0], 0];
      for (let c = 1; c < coefficients.length; c += 1) value = csub(cmul(value, roots[i]), [-coefficients[c], 0]);
      let divisor = [1, 0];
      for (let j = 0; j < degree; j += 1) if (i !== j) divisor = cmul(divisor, csub(roots[i], roots[j]));
      const delta = cdiv(value, divisor);
      roots[i] = csub(roots[i], delta);
      largest = Math.max(largest, Math.hypot(...delta));
    }
    if (largest < 1e-8) break;
  }
  return roots;
}

export function estimateFormants(samples, sampleRate, time = 0) {
  const size = Math.max(512, Math.round(sampleRate * 0.025));
  const frame = frameSamples(samples, time * sampleRate, size);
  for (let i = 0; i < size; i += 1) {
    const previous = i ? frame[i - 1] : 0;
    frame[i] = (frame[i] - 0.97 * previous) * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (size - 1)));
  }
  const coefficients = levinsonDurbin(frame, 12);
  if (!coefficients) return [null, null, null];
  const roots = polynomialRoots(Array.from(coefficients));
  const candidates = roots.filter(([, im]) => im > 0.015).map(([re, im]) => {
    const radius = Math.hypot(re, im);
    const frequency = Math.atan2(im, re) * sampleRate / (2 * Math.PI);
    const bandwidth = -Math.log(Math.max(radius, 1e-6)) * sampleRate / Math.PI;
    return { frequency, bandwidth };
  }).filter(({ frequency, bandwidth }) => frequency >= 90 && frequency <= Math.min(5500, sampleRate / 2) && bandwidth < 500)
    .sort((a, b) => a.frequency - b.frequency);
  const distinct = [];
  for (const candidate of candidates) if (!distinct.some((old) => Math.abs(old.frequency - candidate.frequency) < 80)) distinct.push(candidate);
  return [distinct[0]?.frequency ?? null, distinct[1]?.frequency ?? null, distinct[2]?.frequency ?? null];
}

export function calculateSpectralTilt(samples) {
  let size = 1;
  while (size < samples.length) size <<= 1;
  size = Math.max(size, 2);
  const real = new Float64Array(size);
  const imag = new Float64Array(size);
  for (let i = 0; i < samples.length; i += 1) real[i] = samples[i];
  fft(real, imag);
  const bins = [];
  const db = [];
  for (let i = 0; i < size / 2; i += 1) {
    const magnitude = Math.hypot(real[i], imag[i]);
    if (!(magnitude > 0)) return null;
    bins.push(i);
    db.push(20 * Math.log10(magnitude / 0.0002));
  }
  let maxDb = -Infinity
  let minDb = Infinity
  for (const value of db) {
    if (value > maxDb) maxDb = value
    if (value < minDb) minDb = value
  }
  const range = maxDb - minDb
  if (!(range > 0) || bins.length < 2) return null;
  const scaled = db.map((value) => (value + Math.abs(minDb)) * (bins.length - 1) / range);
  const sumX = bins.reduce((a, b) => a + b, 0);
  const sumY = scaled.reduce((a, b) => a + b, 0);
  const sumXX = bins.reduce((sum, x) => sum + x * x, 0);
  const sumXY = bins.reduce((sum, x, i) => sum + x * scaled[i], 0);
  return (sumXY - sumX * sumY / bins.length) / (sumXX - sumX * sumX / bins.length);
}

export function calculateFrequencyArray(samples, sampleRate, timeStep = 0.01) {
  const rows = [];
  for (let time = 0; time < samples.length / sampleRate; time += timeStep) {
    const pitch = estimatePitch(samples, sampleRate, time);
    const formants = pitch ? estimateFormants(samples, sampleRate, time) : [null, null, null];
    rows.push([pitch, ...formants].map((value) => value == null || !Number.isFinite(value) ? null : Math.fround(value)));
  }
  return rows;
}
