const EPSILON = 1e-10;

export function diagonalTransitionLog(classCount, exponent) {
  if (!Number.isInteger(classCount) || classCount < 1) throw new RangeError('classCount must be a positive integer');
  if (!Number.isFinite(exponent) || exponent < 0) throw new RangeError('exponent must be non-negative');
  const changeCost = -exponent * Math.log(10);
  return Array.from({ length: classCount }, (_, from) =>
    Array.from({ length: classCount }, (_, to) => (from === to ? 0 : changeCost)));
}

export function viterbiDecode(probabilities, transitionLog) {
  if (!Array.isArray(probabilities) || probabilities.length === 0) return [];
  const classes = probabilities[0].length;
  if (!Number.isInteger(classes) || classes < 1 || transitionLog.length !== classes || transitionLog.some((row) => row.length !== classes)) {
    throw new RangeError('emission and transition dimensions do not match');
  }
  const back = Array.from({ length: probabilities.length }, () => new Int32Array(classes));
  let previous = Float64Array.from(probabilities[0], (p) => Math.log(Math.max(EPSILON, p)));
  for (let t = 1; t < probabilities.length; t += 1) {
    if (probabilities[t].length !== classes) throw new RangeError(`frame ${t} has inconsistent class count`);
    const current = new Float64Array(classes);
    for (let to = 0; to < classes; to += 1) {
      let bestFrom = 0;
      let best = previous[0] + transitionLog[0][to];
      for (let from = 1; from < classes; from += 1) {
        const candidate = previous[from] + transitionLog[from][to];
        if (candidate > best) { best = candidate; bestFrom = from; }
      }
      current[to] = best + Math.log(Math.max(EPSILON, probabilities[t][to]));
      back[t][to] = bestFrom;
    }
    previous = current;
  }
  let state = 0;
  for (let i = 1; i < classes; i += 1) if (previous[i] > previous[state]) state = i;
  const path = new Int32Array(probabilities.length);
  path[path.length - 1] = state;
  for (let t = path.length - 1; t > 0; t -= 1) path[t - 1] = back[t][path[t]];
  return [...path];
}

export function labelsToSegments(classIndices, labels, frameStepSeconds = 0.01) {
  if (!Number.isFinite(frameStepSeconds) || frameStepSeconds <= 0) throw new RangeError('frameStepSeconds must be positive');
  if (classIndices.length === 0) return [];
  const segments = [];
  let start = 0;
  let current = classIndices[0];
  for (let frame = 1; frame <= classIndices.length; frame += 1) {
    if (frame < classIndices.length && classIndices[frame] === current) continue;
    if (current < 0 || current >= labels.length) throw new RangeError(`unknown class index ${current}`);
    segments.push({ label: labels[current], start: start * frameStepSeconds, end: frame * frameStepSeconds });
    if (frame < classIndices.length) { current = classIndices[frame]; start = frame; }
  }
  return segments;
}

export function decodeAndMerge(probabilities, labels, { frameStepSeconds = 0.01, transitionExponent = 0 } = {}) {
  if (probabilities.length && probabilities[0].length !== labels.length) throw new RangeError('labels length does not match class count');
  const path = viterbiDecode(probabilities, diagonalTransitionLog(labels.length, transitionExponent));
  return labelsToSegments(path, labels, frameStepSeconds);
}
