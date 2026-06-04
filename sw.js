// Service Worker для Феон - кэширует страницу и работает офлайн

var CACHE_NAME = 'feon-cache-v1';
var urlsToCache = [
  '/chat/',
  '/chat/index.html',
  '/chat/manifest.json'
];

// Установка: кэшируем страницу
self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(CACHE_NAME).then(function(cache) {
      return cache.addAll(urlsToCache);
    })
  );
});

// Перехват запросов: отдаём из кэша, если офлайн
self.addEventListener('fetch', function(event) {
  event.respondWith(
    caches.match(event.request).then(function(response) {
      // Есть в кэше — отдаём
      if (response) {
        return response;
      }
      // Нет в кэше — пробуем сеть
      return fetch(event.request).then(function(networkResponse) {
        // Кэшируем успешные ответы
        if (networkResponse && networkResponse.status === 200) {
          var responseClone = networkResponse.clone();
          caches.open(CACHE_NAME).then(function(cache) {
            cache.put(event.request, responseClone);
          });
        }
        return networkResponse;
      }).catch(function() {
        // Сеть недоступна — отдаём заглушку для некэшированных запросов
        if (event.request.mode === 'navigate') {
          return caches.match('/chat/');
        }
        return new Response('', {status: 408});
      });
    })
  );
});
