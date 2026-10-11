const APP_CACHE = 'voice-tools-app-v1'

self.addEventListener('install', event => {
  event.waitUntil(self.skipWaiting())
})

self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    const names = await caches.keys()
    await Promise.all(names.filter(name => name.startsWith('voice-tools-app-') && name !== APP_CACHE).map(name => caches.delete(name)))
    await self.clients.claim()
  })())
})

self.addEventListener('message', event => {
  if (event.data?.type !== 'DOWNLOAD_MODELS') return
  const port = event.ports[0]
  const operation = (async () => {
    try {
      const cacheName = event.data.cacheName === 'voice-tools-models-v1' ? event.data.cacheName : 'voice-tools-models-v1'
      const urls = Array.isArray(event.data.urls) ? event.data.urls : []
      if (!urls.length) throw new Error('No model files were requested.')
      const safeUrls = urls.map(value => new URL(value, self.registration.scope)).filter(url =>
        url.origin === self.location.origin && url.pathname.includes('/models/tfjs/'))
      if (safeUrls.length !== urls.length) throw new Error('A model URL is outside the trusted same-origin model directory.')
      const cache = await caches.open(cacheName)
      for (let index = 0; index < safeUrls.length; index += 1) {
        const url = safeUrls[index]
        const response = await fetch(url.href, { cache: 'reload' })
        if (!response.ok) throw new Error(`Could not download ${url.pathname} (HTTP ${response.status}).`)
        await cache.put(url.href, response)
        port?.postMessage({ type: 'progress', message: `Downloaded ${index + 1} of ${safeUrls.length} model files…` })
      }
      port?.postMessage({ type: 'complete' })
    } catch (error) {
      port?.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) })
    }
  })()
  event.waitUntil(operation)
})

self.addEventListener('fetch', event => {
  const request = event.request
  if (request.method !== 'GET') return
  const url = new URL(request.url)
  if (url.origin !== self.location.origin) return

  if (url.pathname.includes('/models/tfjs/')) {
    event.respondWith((async () => {
      const cached = await caches.match(request, { cacheName: 'voice-tools-models-v1' })
      if (cached) return cached
      const response = await fetch(request)
      if (response.ok) {
        const cache = await caches.open('voice-tools-models-v1')
        await cache.put(request, response.clone())
      }
      return response
    })())
    return
  }

  if (!['document', 'script', 'style', 'image', 'font', 'worker'].includes(request.destination)) return
  event.respondWith((async () => {
    const cache = await caches.open(APP_CACHE)
    try {
      const response = await fetch(request)
      if (response.ok && response.type === 'basic') await cache.put(request, response.clone())
      return response
    } catch {
      const cached = await cache.match(request)
      if (cached) return cached
      if (request.mode === 'navigate') {
        const appShell = await cache.match(new URL('index.html', self.registration.scope).href)
        if (appShell) return appShell
      }
      return Response.error()
    }
  })())
})
