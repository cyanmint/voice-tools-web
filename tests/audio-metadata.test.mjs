import test from 'node:test'
import assert from 'node:assert/strict'
import { detectAudioSampleRate } from '../src/js/local-inference/audio-metadata.js'

function adtsHeader(sampleRateIndex) {
  return Uint8Array.from([0xff, 0xf1, 0x40 | (sampleRateIndex << 2), 0x80, 0, 0, 0])
}

function wavHeader(sampleRate) {
  const bytes = new Uint8Array(44)
  const view = new DataView(bytes.buffer)
  const writeAscii = (offset, value) => [...value].forEach((character, index) => { bytes[offset + index] = character.charCodeAt(0) })
  writeAscii(0, 'RIFF')
  view.setUint32(4, 36, true)
  writeAscii(8, 'WAVE')
  writeAscii(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeAscii(36, 'data')
  return bytes
}

test('detectAudioSampleRate reads the encoded AAC ADTS sample rate', () => {
  assert.equal(detectAudioSampleRate(adtsHeader(4)), 44100)
  assert.equal(detectAudioSampleRate(adtsHeader(3)), 48000)
})

test('detectAudioSampleRate recognizes WebM Opus as 48 kHz', () => {
  const header = new Uint8Array([0x1a, 0x45, 0xdf, 0xa3, ...Array.from('A_OPUS', character => character.charCodeAt(0))])
  assert.equal(detectAudioSampleRate(header), 48000)
})

test('detectAudioSampleRate reads the source rate from a WAV fmt chunk', () => {
  assert.equal(detectAudioSampleRate(wavHeader(16000)), 16000)
})

test('detectAudioSampleRate returns null for unknown or invalid headers', () => {
  assert.equal(detectAudioSampleRate(Uint8Array.from([1, 2, 3, 4])), null)
  assert.equal(detectAudioSampleRate(adtsHeader(15)), null)
})
