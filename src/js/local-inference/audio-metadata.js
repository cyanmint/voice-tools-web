const AAC_SAMPLE_RATES = [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350]

const asciiAt = (bytes, offset, text) => {
  if (offset < 0 || offset + text.length > bytes.length) return false
  for (let index = 0; index < text.length; index += 1) {
    if (bytes[offset + index] !== text.charCodeAt(index)) return false
  }
  return true
}

function wavSampleRate(bytes) {
  if (!asciiAt(bytes, 0, 'RIFF') || !asciiAt(bytes, 8, 'WAVE')) return null
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  for (let offset = 12; offset + 8 <= bytes.length;) {
    const size = view.getUint32(offset + 4, true)
    if (asciiAt(bytes, offset, 'fmt ') && size >= 16 && offset + 28 <= bytes.length) {
      const rate = view.getUint32(offset + 12, true)
      return rate >= 8000 && rate <= 192000 ? rate : null
    }
    if (size > bytes.length - offset - 8) break
    offset += 8 + size + (size & 1)
  }
  return null
}

function adtsSampleRate(bytes) {
  const limit = Math.min(bytes.length - 7, 65536)
  for (let offset = 0; offset <= limit; offset += 1) {
    if (bytes[offset] !== 0xff || (bytes[offset + 1] & 0xf6) !== 0xf0) continue
    const index = (bytes[offset + 2] >> 2) & 0x0f
    const rate = AAC_SAMPLE_RATES[index]
    if (rate) return rate
  }
  return null
}

function mp3SampleRate(bytes) {
  let start = 0
  if (asciiAt(bytes, 0, 'ID3') && bytes.length >= 10) {
    const tagSize = ((bytes[6] & 0x7f) << 21) | ((bytes[7] & 0x7f) << 14) | ((bytes[8] & 0x7f) << 7) | (bytes[9] & 0x7f)
    start = 10 + tagSize + ((bytes[5] & 0x10) ? 10 : 0)
  }
  const limit = Math.min(bytes.length - 4, start + 65536)
  for (let offset = start; offset <= limit; offset += 1) {
    if (bytes[offset] !== 0xff || (bytes[offset + 1] & 0xe0) !== 0xe0) continue
    const version = (bytes[offset + 1] >> 3) & 0x03
    const layer = (bytes[offset + 1] >> 1) & 0x03
    const index = (bytes[offset + 2] >> 2) & 0x03
    if (version === 1 || layer === 0 || index === 3) continue
    const base = [44100, 48000, 32000][index]
    return version === 3 ? base : version === 2 ? base / 2 : base / 4
  }
  return null
}

function mp4SampleRate(bytes) {
  const limit = Math.min(bytes.length - 4, 1024 * 1024)
  for (let typeOffset = 4; typeOffset <= limit; typeOffset += 1) {
    if (!asciiAt(bytes, typeOffset, 'mp4a')) continue
    const boxOffset = typeOffset - 4
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    const size = view.getUint32(boxOffset, false)
    const headerSize = size === 1 ? 16 : 8
    const rateOffset = boxOffset + headerSize + 24
    if (rateOffset + 4 > bytes.length) continue
    const rate = view.getUint32(rateOffset, false) >>> 16
    if (rate >= 8000 && rate <= 192000) return rate
  }
  return null
}

function flacSampleRate(bytes) {
  if (!asciiAt(bytes, 0, 'fLaC') || bytes.length < 21 || (bytes[4] & 0x7f) !== 0) return null
  const rate = (bytes[18] << 12) | (bytes[19] << 4) | (bytes[20] >> 4)
  return rate >= 8000 && rate <= 192000 ? rate : null
}

function oggSampleRate(bytes) {
  if (!asciiAt(bytes, 0, 'OggS')) return null
  for (let offset = 0; offset + 16 <= Math.min(bytes.length, 65536); offset += 1) {
    if (asciiAt(bytes, offset, 'OpusHead')) return 48000
    if (bytes[offset] === 1 && asciiAt(bytes, offset + 1, 'vorbis')) {
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
      const rate = view.getUint32(offset + 12, true)
      return rate >= 8000 && rate <= 192000 ? rate : null
    }
  }
  return null
}

function webmSampleRate(bytes) {
  if (bytes.length < 4 || bytes[0] !== 0x1a || bytes[1] !== 0x45 || bytes[2] !== 0xdf || bytes[3] !== 0xa3) return null
  for (let offset = 4; offset + 6 <= Math.min(bytes.length, 1024 * 1024); offset += 1) {
    if (asciiAt(bytes, offset, 'A_OPUS')) return 48000
  }
  return null
}

/** Detect the encoded audio rate so decodeAudioData does not resample to the output device rate. */
export function detectAudioSampleRate(input) {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input)
  return wavSampleRate(bytes)
    ?? adtsSampleRate(bytes)
    ?? mp4SampleRate(bytes)
    ?? flacSampleRate(bytes)
    ?? oggSampleRate(bytes)
    ?? webmSampleRate(bytes)
    ?? mp3SampleRate(bytes)
}
