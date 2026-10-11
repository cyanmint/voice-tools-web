<template>
  <div class="results">
<!--    <Waveform :audio="audio" v-if="audio" />-->

    <div v-if="analysisLoaded" class="inference-note">
      Analysis and audio processing ran in this browser; no audio was sent to a server.
      <span v-if="!mlEnabled">ML classification is disabled.</span>
      <span v-else>Frame classification used {{ inferenceBackend ? inferenceBackend.toUpperCase() : 'the local TensorFlow.js runtime' }}.</span>
      <span v-if="mlError"> {{ mlError }}</span>
    </div>

    <div class="result-nav unselectable anim">
      <div class="tab-button spectrogram" @click="() => activeTab = 1" :class="{sel: activeTab === 1}">
        Spectrogram
      </div>
      <div class="tab-button" @click="() => activeTab = 2" :class="{sel: activeTab === 2}">
        Classification
      </div>
      <div class="tab-button" @click="() => activeTab = 3" :class="{sel: activeTab === 3}">
        Position
      </div>
    </div>
    <div v-if="loadError" class="load-error" role="alert">{{ loadError }}</div>

    <!-- Spectrogram View -->
    <div class="result-tab card shadow" :style="activeTab === 1 ? {} : {display: 'none'}">
      <Spectrogram v-if="spec?.length" :spec="spec" :sr="specSr" :freq-arrays="freqArrays"/>
      <div v-else-if="loadError" class="load-placeholder">{{ loadError }}</div>
      <Loading v-else />
    </div>

    <!-- Classification Results -->
    <div class="result-tab card shadow" v-if="activeTab === 2">
      <ClassificationResults v-if="stats" :stats="stats" :ml="ml" :ml-enabled="mlEnabled" :audio-duration="audioDuration"/>
      <div v-else-if="loadError" class="load-placeholder">{{ loadError }}</div>
      <Loading v-else />
    </div>

    <!-- Pitch vs Formant -->
    <div class="result-tab card shadow" v-if="activeTab === 3">
      <PitchResGraph v-if="stats" :stats="stats" />
      <div v-else-if="loadError" class="load-placeholder">{{ loadError }}</div>
      <Loading v-else />
    </div>
  </div>
</template>

<script lang="ts">
import {Options, Vue} from 'vue-class-component';
import Spectrogram from "@/views/comp/Spectrogram.vue";
import Waveform from "@/views/comp/Waveform.vue";
import PitchResGraph from "@/views/comp/PitchResGraph.vue";
import ClassificationResults, {MLFrame, StatsResult} from "@/views/comp/ClassificationResults.vue";
import Loading from '@/views/comp/Loading.vue';
import {getLocalAnalysis} from '@/js/local-inference/LocalResultStore';
import { Prop } from 'vue-property-decorator';

@Options({components: {PitchResGraph, ClassificationResults, Waveform, Spectrogram, Loading}})
export default class Results extends Vue
{
  @Prop({required: true}) uuid!: string

  activeTab = 1
  stats: StatsResult = null
  ml: MLFrame[] = []
  spec: Float32Array[] = null
  specSr: number
  freqArrays: {[index: string]: Float32Array} = null
  audioDuration: number
  mlEnabled = false
  inferenceBackend = ''
  mlError = ''
  analysisLoaded = false
  loadError = ''

  async created() {
    try {
      const result = await getLocalAnalysis(this.uuid)
      if (!result) throw new Error('This local result is no longer available in this browser. Analyze the audio again.')
      this.stats = result.result as unknown as StatsResult
      this.ml = result.ml as MLFrame[]
      this.spec = result.spec
      this.specSr = result.spec_sr
      this.freqArrays = result.freqArrays
      this.audioDuration = result.audio_dur
      this.mlEnabled = result.ml_enabled
      this.inferenceBackend = result.inference_backend || ''
      this.mlError = result.ml_error || ''
      this.analysisLoaded = true
    } catch (error) {
      this.loadError = error instanceof Error ? error.message : String(error)
      console.error('Could not load browser-local result:', error)
    }
  }
}
</script>

<style lang="sass" scoped>
@import "src/css/colors"

.results
  margin: 0 20px 0
  position: relative

.results, .result-tab
  display: flex
  flex-direction: column
  flex: 1
  min-height: 0

// Tab content
.result-tab
  border-top: none
  margin-top: -1px
  overflow-x: hidden

  // Card effect
  margin-bottom: 20px
  z-index: 2

.inference-note
  position: absolute
  top: 0
  left: 0
  right: 0
  height: 20px
  margin: 0 20px
  color: $color-fg
  font-size: 12px
  line-height: 20px
  text-align: center
  white-space: nowrap
  z-index: 1

.load-error, .load-placeholder
  margin: 24px
  color: #ffbc29
  text-align: center
  overflow-wrap: anywhere

.result-nav
  display: flex
  flex-direction: row
  margin-top: 20px
  justify-content: center
  min-height: 40px
  color: $color-text-main
  gap: 15px

  .tab-button
    padding: 5px 10px
    display: flex
    flex-direction: column
    justify-content: center
    border-radius: 10px 10px 0 0
    background: lighten(#fff4eb, 1)
    min-width: 115px

  .tab-button.sel
    border-radius: 10px 10px 0 0
    min-height: 30px
    margin-top: -10px
    background: white
    color: #7c7c7c

    // Show shadow without the bottom side
    box-shadow: 0 2px 12px -2px rgb(0 0 0 / 10%)
    clip-path: inset(-5px -5px 0 -5px)
    z-index: 3

  .tab-button.spectrogram
    background: #232323
    color: #c9c9c9

.result-nav::before, .result-nav::after
  content: ''
  flex-grow: 1

@media only screen and (max-width: 530px)
  .result-nav
    gap: 10px

    .tab-button
        font-size: 0.9em
        min-width: 80px

@media only screen and (max-width: 430px)
  .result-nav
    gap: 10px

    .tab-button
        font-size: 0.9em
        min-width: 80px

  // Disable card view for smaller devices
  .results
    margin: 0
    font-size: 0.9em

  .result-tab
    margin: 0
    border-radius: 0
</style>
