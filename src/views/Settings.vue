<template>
  <div id="settings" class="fbox-v">
    <div class="card shadow pad">
      <div id="text">
        <div id="title">{{ $t("settings.title") }}</div>
        <div id="subtitle">You can toggle things here!</div>
      </div>

      <section class="local-models">
        <div><strong>Local inference models</strong></div>
        <p>Download the required TensorFlow.js weights into this browser. The files are cached by the local service worker for offline use.</p>
        <el-button type="primary" :loading="modelsBusy" @click="downloadLocalModels">{{ modelsBusy ? 'Downloading…' : 'Download models' }}</el-button>
        <div class="model-status" role="status">{{ modelStatus }}</div>
      </section>

      <div id="options">
        <div v-for="option of options" :class="getClass(option)">
          <div class="desc" v-html="marked(option.desc)" />

          <div v-if="option.type === 'boolean'">
            <el-switch v-model="option.val" active-color="pink"/>
          </div>

          <div v-if="option.type === 'string'">
            <hy-input v-model="option.val" />
          </div>

          <div class="modified-text" v-if="option.modified">
            <span>Modified!</span> <span>Default is <code>{{option.def}}</code></span>
          </div>
        </div>
      </div>

      <el-button type="danger" plain round @click="reset">Reset</el-button>
    </div>
  </div>
</template>

<script lang="ts">
import {options, Setting} from '@/js/Setting';
import {Options, Vue} from 'vue-class-component';
import {marked} from 'marked';
import HyInput from "@/views/comp/HyInput.vue";
import {ElMessage, ElMessageBox} from "element-plus";
import {ref} from "vue";
import {downloadModels as downloadModelFiles, modelsDownloaded} from '@/js/local-inference/ModelManager';

@Options({components: {HyInput}})
export default class Settings extends Vue
{
  declare $t: (key: string) => string

  // options = options.map(it => ref(it))
  options = options
  marked = marked
  modelsBusy = false
  modelStatus = 'Checking local model cache…'

  async mounted() {
    try {
      this.modelStatus = await modelsDownloaded() ? 'Both models are downloaded and ready for offline use.' : 'Models are not downloaded yet.'
    } catch (error) {
      this.modelStatus = error instanceof Error ? error.message : String(error)
    }
  }

  async downloadLocalModels() {
    this.modelsBusy = true
    try {
      await downloadModelFiles(message => { this.modelStatus = message })
      this.modelStatus = 'Both models are downloaded and ready for offline use.'
      ElMessage.success('Local inference models downloaded.')
    } catch (error) {
      this.modelStatus = error instanceof Error ? error.message : String(error)
      ElMessage.error(this.modelStatus)
    } finally {
      this.modelsBusy = false
    }
  }

  getClass<T>(option: Setting<T>)
  {
    const cls = {'modified': option.modified}
    cls[option.type] = true
    return cls
  }

  reset()
  {
    ElMessageBox.confirm('Do you really want to reset?')
        .then(() => {
          options.forEach(it => it.reset())
          this.$forceUpdate()
        })
  }
}
</script>

<style lang="sass" scoped>
#settings
  margin: 20px

  .card
    width: 100%
    max-width: 600px
    margin: 0 auto
    display: flex
    flex-direction: column
    gap: 20px

  #text
    #title
      font-size: 1.2em

    #subtitle
      font-size: 0.8em

  .local-models
    padding: 12px
    border: 1px solid #ddd
    border-radius: 6px
    text-align: left

    p
      margin: 8px 0 12px
      font-size: 0.9em

  .model-status
    margin-top: 8px
    overflow-wrap: anywhere

  #options
    text-align: left
    .desc
      margin-bottom: 0
      font-size: 0.9em

  .modified-text
    margin-top: 5px
    font-size: 0.8em
    color: #9f9f9f

    code
      padding: 0.2em 0.4em
      margin: 0
      background-color: rgb(175 184 193 / 20%)
      border-radius: 6px

    span:first-child
      color: red
</style>

<style lang="sass">
#Settings #options .desc
  p
    margin-bottom: 5px
</style>
