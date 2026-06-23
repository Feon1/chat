(function() {
    "use strict";

    (function() {
        var i;
        const o = "loader.js";
        const l = "botman-webchat.D6vXXlSJ.js";
        const a = window;

        // Если виджет уже загружен — выходим
        if ((i = a.BotmanWebChat) != null && i._loaded) return;

        // Методы, которые будут доступны до загрузки бандла
        const f = [
            "connect", "auth", "identify", "open", "close",
            "sendMessage", "track", "onReady", "addCallback",
            "removeCallback", "destroy"
        ];

        const c = [];
        const s = {
            _queue: c,
            _loaded: false
        };

        // Создаём заглушки методов, складывающие вызовы в очередь
        for (const e of f) {
            s[e] = function(...t) {
                c.push([e, t]);
            };
        }

        a.BotmanWebChat = s;

        // Функция: получить базовый путь к загрузчику
        function b() {
            const e = document.getElementsByTagName("script");
            for (let t = e.length - 1; t >= 0; t--) {
                const n = e[t].src;
                if (n && n.includes(o)) {
                    return n.substring(0, n.lastIndexOf("/") + 1);
                }
            }
            return "";
        }

        // Функция: получить сам тег загрузчика
        function m() {
            const e = document.getElementsByTagName("script");
            for (let t = e.length - 1; t >= 0; t--) {
                if (e[t].src && e[t].src.includes(o)) {
                    return e[t];
                }
            }
            return null;
        }

        // Основная функция: загрузка бандла
        function r() {
            const e = b();
            const t = m();
            const n = document.createElement("script");

            // ЯВНО УКАЗЫВАЕМ UTF-8 ДЛЯ ПОДГРУЖАЕМОГО БАНДЛА
            n.charset = "utf-8";

            n.type = "module";
            n.src = e + l;
            n.async = true;

            if (t) {
                const d = t.getAttribute("data-webchat-id");
                const u = t.getAttribute("data-webchat-api");
                if (d) n.setAttribute("data-webchat-id", d);
                if (u) n.setAttribute("data-webchat-api", u);
            }

            n.onerror = function() {
                console.error("[BotmanWebChat] Failed to load widget bundle");
            };

            document.head.appendChild(n);
        }

        // Запускаем после полной загрузки DOM
        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", r);
        } else {
            r();
        }

    })();
})();
