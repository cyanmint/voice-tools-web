export function resolveModelUrls(baseUrl, origin) {
  const root = new URL(baseUrl, origin)
  return {
    vad: new URL('models/tfjs/keras_speech_music_noise_cnn/model.json', root).href,
    gender: new URL('models/tfjs/keras_male_female_cnn/model.json', root).href,
  }
}
