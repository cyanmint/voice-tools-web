<template>
  <div id="home" ref="el" class="fbox-v">
    <div class="usage">
      Welcome to the voice analysis tool!
      <br>
      Audio decoding and analysis run in this browser. Your audio file is never uploaded to an analysis server.
      <br>
      ML classification can be switched off in Settings; download the models there once to enable offline inference.
    </div>

    <div class="model-status" role="status">{{ modelStatus }}</div>

    <div class="drop-box unselectable shadow card pad clickable" :class="{busy: dropped}" @dragover="(e) => e.preventDefault()" @drop="onDrop" @click="chooseAudio">
      <span class="drop-label" v-if="!dropped">Drop audio here or choose a file</span>
      <span class="drop-label" v-else>{{ progressText }}</span>
    </div>
    <input ref="audioPicker" class="visually-hidden" type="file" accept="audio/*,.wav,.mp3,.ogg,.m4a,.flac,.webm" @change="onAudioChosen">

    <div class="drop-box unselectable shadow card pad record clickable" @click="btnRecord" v-if="recorderCompatible">
      <span class="drop-label" v-if="!recorder">Record audio</span>
      <span class="drop-label" v-else>Stop recording</span>
    </div>
  </div>
</template>

<script lang="ts">
import {ElMessage} from "element-plus";
import {Options, Vue} from "vue-class-component";
import {getSetting} from '@/js/Setting';
import {runLocalAnalysis} from '@/js/local-inference/LocalAnalysisClient';
import {modelsDownloaded} from '@/js/local-inference/ModelManager';
import {saveLocalAnalysis} from '@/js/local-inference/LocalResultStore';
import {detectAudioSampleRate} from '@/js/local-inference/audio-metadata.js';
import router from '@/router';

function downmix(buffer: AudioBuffer): Float32Array {
  const output = new Float32Array(buffer.length)
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    const data = buffer.getChannelData(channel)
    for (let index = 0; index < data.length; index += 1) output[index] += data[index] / buffer.numberOfChannels
  }
  return output
}

function newAnalysisId(): string {
  return typeof crypto.randomUUID === 'function' ? crypto.randomUUID() : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
}

@Options({})
export default class Home extends Vue
{
  declare $refs: {
    audioPicker: HTMLInputElement
  }

  dropped = false
  progressText = 'Preparing local analysis…'
  modelStatus = 'Checking local model cache…'
  recorderCompatible = !!navigator.mediaDevices?.getUserMedia && typeof MediaRecorder !== 'undefined'
  recorder: MediaRecorder = null
  recordChunks: Blob[] = []
  recordingStream: MediaStream = null

  async mounted() {
    if (!getSetting('local.inference.enabled').val) {
      this.modelStatus = 'ML classification is disabled. Voice statistics and the spectrogram still run locally.'
      return
    }
    try {
      this.modelStatus = await modelsDownloaded()
        ? 'Local models are cached. Audio stays on this device during analysis.'
        : 'Models are not downloaded. Visit Settings to download them, or disable ML classification there.'
    } catch (error) {
      this.modelStatus = error instanceof Error ? error.message : String(error)
    }
  }

  chooseAudio() {
    if (!this.dropped) this.$refs.audioPicker?.click()
  }

  async onAudioChosen(event: Event) {
    const input = event.target as HTMLInputElement
    const file = input.files?.[0]
    input.value = ''
    if (file) await this.loadFile(file)
  }

  async loadFile(file: File) {
    if (this.dropped) return
    this.dropped = true
    let context: AudioContext | null = null
    try {
      this.progressText = 'Decoding audio locally…'
      const audioData = await file.arrayBuffer()
      const sourceSampleRate = detectAudioSampleRate(new Uint8Array(audioData))
      context = sourceSampleRate
        ? new AudioContext({sampleRate: sourceSampleRate})
        : new AudioContext()
      const decoded = await context.decodeAudioData(audioData)
      if (sourceSampleRate && decoded.sampleRate !== sourceSampleRate) {
        throw new Error(`Could not preserve the source audio rate (${sourceSampleRate} Hz); decoded at ${decoded.sampleRate} Hz.`)
      }
      const inferenceEnabled = Boolean(getSetting('local.inference.enabled').val)
      if (inferenceEnabled && !await modelsDownloaded()) {
        throw new Error('Download the TensorFlow.js models in Settings first, or turn off ML classification.')
      }
      this.progressText = 'Analyzing audio locally…'
      const analysis = await runLocalAnalysis(
        downmix(decoded),
        decoded.sampleRate,
        decoded.duration,
        inferenceEnabled,
        message => { this.progressText = message },
      )
      const id = newAnalysisId()
      await saveLocalAnalysis(id, analysis)
      this.progressText = 'Analysis complete. Opening results…'
      await router.push({name: 'Results', params: {uuid: id}})
    } catch (error) {
      this.dropped = false
      this.progressText = 'Drop audio here or choose a file'
      const message = error instanceof Error ? error.message : String(error)
      ElMessage.error(message)
      console.error('Browser-local audio analysis failed:', error)
    } finally {
      if (context && context.state !== 'closed') await context.close().catch(() => undefined)
    }
  }

  async onDrop(event: DragEvent) {
    event.preventDefault()
    const files = Array.from(event.dataTransfer?.files || [])
    if (files.length !== 1) {
      ElMessage.error(`Choose exactly one audio file (received ${files.length}).`)
      return
    }
    await this.loadFile(files[0])
  }

  async btnRecord() {
    if (this.recorder) {
      const recorder = this.recorder
      recorder.stop()
      this.recorder = null
      return
    }
    try {
      this.recordingStream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1}})
      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'].find(type => MediaRecorder.isTypeSupported(type))
      const recorder = mimeType ? new MediaRecorder(this.recordingStream, {mimeType}) : new MediaRecorder(this.recordingStream)
      this.recorder = recorder
      this.recordChunks = []
      recorder.addEventListener('dataavailable', event => { if (event.data.size) this.recordChunks.push(event.data) })
      recorder.addEventListener('stop', () => {
        const blob = new Blob(this.recordChunks, {type: recorder.mimeType || 'audio/webm'})
        const file = new File([blob], 'recorded-audio', {type: blob.type})
        this.recordingStream?.getTracks().forEach(track => track.stop())
        this.recordingStream = null
        this.loadFile(file)
      }, {once: true})
      recorder.start()
    } catch (error) {
      this.recordingStream?.getTracks().forEach(track => track.stop())
      this.recordingStream = null
      const message = error instanceof Error ? error.message : String(error)
      ElMessage.error(`Could not start microphone recording: ${message}`)
    }
  }
}
</script>

<style lang="sass" scoped>
@import "src/css/colors"

#home
  margin: 0 20px 0
  gap: 20px

  .usage
    color: $color-fg

  .model-status
    color: $color-fg
    overflow-wrap: anywhere

  .visually-hidden
    position: absolute
    width: 1px
    height: 1px
    padding: 0
    margin: -1px
    overflow: hidden
    clip: rect(0, 0, 0, 0)
    white-space: nowrap
    border: 0

  .drop-box
    width: 100%
    max-width: 600px
    margin: 0 auto
    min-height: 160px
    display: flex
    align-items: center
    justify-content: center

    .drop-label
      color: $color-bg
      font-size: 1.3em
      text-align: center

  .drop-box.record
    min-height: 90px

  .drop-box.busy
    cursor: progress
</style>
