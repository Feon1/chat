import aiohttp
import os
import re
import json
import asyncio
import uuid
import random
import traceback
import hashlib
import ssl
import certifi
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models
from bs4 import BeautifulSoup


# Импорт для Object Storage (если используется)
try:
    import boto3
    from botocore.exceptions import ClientError
    from botocore.config import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    print("⚠️ boto3 не установлен. Функции Object Storage недоступны.")

load_dotenv()

app = FastAPI(title="Feon RAG Adapter (Telegram + Web)")

# Разрешаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://feon1.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ И НАСТРОЙКИ
# ==========================================

YANDEX_FUNCTION_URL = os.getenv(
    "YANDEX_FUNCTION_URL",
    "https://d5dq57ou1bsu78horuke.avjje9e3.apigw.yandexcloud.net/query"
)
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")

if not YANDEX_API_KEY:
    print("⚠️ ВНИМАНИЕ: YANDEX_API_KEY не задан. Вызовы Yandex Cloud будут падать.")

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не задан в переменных окружения")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")
POLZA_API_KEY = os.getenv("POLZA_API_KEY")

COLLECTION_NAME = "xiaozhi_knowledge_tg"
HISTORY_COLLECTION = "chat_history_tg"

JINA_API_URL = "https://api.jina.ai/v1/embeddings"

# Настройки Object Storage (опционально)
BUCKET_NAME = os.getenv("BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ru-central1")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "https://storage.yandexcloud.net")

# Инициализация клиентов
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

if BOTO3_AVAILABLE and all([BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
    s3_client = boto3.client(
        's3',
        endpoint_url=AWS_ENDPOINT_URL,
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(connect_timeout=5, read_timeout=10)
    )
    print("✅ S3 клиент инициализирован")
else:
    s3_client = None

# Системный промпт
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Ты — Феон, верующий ИИ.")
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read().strip()
        print(f"✅ Системный промпт загружен из файла")
except FileNotFoundError:
    print("ℹ️ Используется SYSTEM_PROMPT из переменной окружения")


# SSL-контекст (оставлен для совместимости)
_ssl_context = ssl.create_default_context(cafile=certifi.where())

# Маркер, который говорит Yandex-функции «это уже распарсенная ссылка,
# не уходи в режим накопления длинных текстов»
RENDER_LINK_MARKER = "[RENDER_LINK]"


def verify_admin(request: Request):
    token = request.headers.get("x-admin-token")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Неверный токен администратора")


# ==========================================
# 🕷️ ИЗВЛЕЧЕНИЕ КОНТЕНТА ИЗ ССЫЛОК (НА RENDER)
# ==========================================

async def fetch_telegram_post(url: str) -> str:
    """
    Извлекает текст КОНКРЕТНОГО поста из публичного Telegram-канала.
    Поддерживает форматы:
      https://t.me/channel/123
      https://t.me/s/channel/123
      https://telegram.me/channel/123
      ... + любые query-параметры (?single, ?embed и т.п.)
    Приватные каналы (t.me/c/...) не поддерживаются.
    """
    print(f"📱 [TG] Обрабатываем ссылку: {url}")

    # Убираем query-параметры, хвостовые слеши
    url_clean = url.split("?")[0].split("#")[0].rstrip("/")

    match = re.search(
        r'(?:t\.me|telegram\.me)/(?:s/)?([^/]+)/(\d+)',
        url_clean
    )
    if not match:
        print(f"⚠️ [TG] Не удалось распознать URL: {url}")
        return ""

    channel, post_id = match.group(1), match.group(2)

    if channel == "c":
        print("⚠️ [TG] Приватный канал — содержимое недоступно")
        return ""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    # Пробуем два варианта — превью канала и embed-виджет
    candidate_urls = [
        f"https://t.me/s/{channel}/{post_id}",
        f"https://t.me/{channel}/{post_id}?embed=1&mode=tme",
    ]

    async with httpx.AsyncClient(
        timeout=20.0,
        follow_redirects=True,
        headers=headers,
        verify=False,
    ) as client:
        for web_url in candidate_urls:
            try:
                print(f"🌐 [TG] Загружаем: {web_url}")
                response = await client.get(web_url)
                if response.status_code != 200:
                    print(f"⚠️ [TG] HTTP {response.status_code} от {web_url}")
                    continue

                soup = BeautifulSoup(response.text, "lxml")

                target = f"{channel}/{post_id}"
                collected = []

                # 1) Ищем ИМЕННО нужный пост по data-post
                for post in soup.select(".tgme_widget_message"):
                    data_post = post.get("data-post", "")
                    if data_post == target or data_post.endswith(f"/{post_id}"):
                        # Основной текст
                        txt_el = post.select_one(".tgme_widget_message_text")
                        if txt_el:
                            collected.append(txt_el.get_text(separator="\n", strip=True))
                        # Подпись к фото/видео/док
                        cap_el = post.select_one(".tgme_widget_message_caption")
                        if cap_el:
                            collected.append(cap_el.get_text(separator="\n", strip=True))
                        break

                # 2) Fallback — если структура поменялась, берём первый текстовый блок
                if not collected:
                    for el in soup.select(
                        ".tgme_widget_message_text, .tgme_widget_message_caption"
                    ):
                        txt = el.get_text(separator="\n", strip=True)
                        if txt:
                            collected.append(txt)
                            break  # только первый, чтобы не склеивать соседние посты

                if not collected:
                    print("⚠️ [TG] Текст поста не найден в HTML")
                    continue

                result = "\n\n".join(collected)
                result = re.sub(r'\n{3,}', '\n\n', result).strip()

                MAX_LEN = 5000
                if len(result) > MAX_LEN:
                    result = result[:MAX_LEN] + "...\n[Текст обрезан]"

                print(f"✅ [TG] Извлечено {len(result)} символов из {channel}/{post_id}")
                return result

            except Exception as e:
                print(f"⚠️ [TG] Ошибка загрузки {web_url}: {e}")
                continue

    print("❌ [TG] Все попытки извлечь пост провалились")
    return ""


async def fetch_vk_post(url: str) -> str:
    """Извлекает текст поста ВКонтакте через API."""
    match = re.search(r'wall(-?\d+)_(\d+)', url)
    if not match:
        print(f"⚠️ [VK] Не удалось распознать URL: {url}")
        return ""

    owner_id, post_id = match.group(1), match.group(2)
    vk_token = os.getenv("VK_USER_TOKEN") or os.getenv("VK_GROUP_TOKEN")
    if not vk_token:
        print("⚠️ [VK] Нет VK_USER_TOKEN или VK_GROUP_TOKEN")
        return ""

    async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
        # 1) wall.getById
        try:
            resp = await client.get(
                "https://api.vk.com/method/wall.getById",
                params={"posts": f"{owner_id}_{post_id}", "access_token": vk_token, "v": "5.199"},
            )
            data = resp.json()
            if "error" not in data and data.get("response"):
                text = data["response"][0].get("text", "")
                if text:
                    print(f"✅ [VK] wall.getById: {len(text)} симв.")
                    return text
        except Exception as e:
            print(f"⚠️ [VK] wall.getById ошибка: {e}")

        # 2) wall.get
        try:
            resp = await client.get(
                "https://api.vk.com/method/wall.get",
                params={"owner_id": owner_id, "count": 100, "access_token": vk_token, "v": "5.199"},
            )
            data = resp.json()
            if "error" not in data:
                items = data.get("response", {}).get("items", [])
                for post in items:
                    if str(post.get("id")) == post_id:
                        text = post.get("text", "")
                        if text:
                            print(f"✅ [VK] wall.get: {len(text)} симв.")
                            return text
        except Exception as e:
            print(f"⚠️ [VK] wall.get ошибка: {e}")

    return ""


async def fetch_url_content(url: str) -> str:
    """Извлекает основной текст со страницы по URL."""
    print(f"🌐 [FETCH] Загружаем: {url}")

    # ===== Telegram =====
    if "t.me/" in url or "telegram.me/" in url:
        tg_text = await fetch_telegram_post(url)
        if tg_text:
            return tg_text
        print("⚠️ [TG] Не удалось получить пост")

    # ===== VK =====
    if "vk.ru/wall" in url or "vk.com/wall" in url:
        vk_text = await fetch_vk_post(url)
        if vk_text and len(vk_text) > 100:
            return vk_text
        print("⚠️ [VK] Не удалось получить пост через API, пробуем обычную загрузку")

    # ===== Обычный сайт =====
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers=headers,
            verify=False,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            print(f"✅ [FETCH] HTTP {response.status_code}, длина {len(response.text)}")

            soup = BeautifulSoup(response.text, "lxml")

            for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                element.decompose()

            text = soup.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            cleaned = "\n".join(lines)

            if len(cleaned) < 300:
                print(f"⚠️ [FETCH] Текст слишком короткий ({len(cleaned)})")
                return ""

            letters = sum(c.isalpha() for c in cleaned)
            if letters / len(cleaned) < 0.5:
                print(f"⚠️ [FETCH] Мало букв ({letters}/{len(cleaned)})")
                return ""

            ui_markers = ["поделиться", "вернуться к странице", "показать список", "посты сообщества"]
            if any(word in cleaned.lower() for word in ui_markers):
                print("⚠️ [FETCH] Обнаружен UI-мусор")
                return ""

            MAX_LEN = 5000
            if len(cleaned) > MAX_LEN:
                cleaned = cleaned[:MAX_LEN] + "...\n[Текст обрезан]"

            print(f"✅ [FETCH] Извлечено {len(cleaned)} символов")
            return cleaned

    except Exception as e:
        print(f"❌ [FETCH] Ошибка загрузки {url}: {e}")
        return ""


def extract_urls(text: str) -> list[str]:
    """Извлекает все ссылки из текста."""
    return re.findall(r'https?://[^\s]+', text)


# ==========================================
# ИНИЦИАЛИЗАЦИЯ ПРИ СТАРТЕ
# ==========================================
@app.on_event("startup")
async def startup_event():
    """Создаем коллекции и индексы при запуске"""

    # 1. Коллекция для базы знаний (Jina dim=384)
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        if info.config.params.vectors.size != 384:
            print(f"⚠️ Размерность {info.config.params.vectors.size} != 384, пересоздаем...")
            qdrant.delete_collection(COLLECTION_NAME)
            raise Exception("Recreate")
        print(f"✅ Коллекция '{COLLECTION_NAME}' найдена (dim=384)")
    except Exception:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )
        print(f"✅ Коллекция '{COLLECTION_NAME}' создана (dim=384)")

    # 2. Коллекция для истории чатов
    try:
        qdrant.get_collection(HISTORY_COLLECTION)
        print(f"✅ Коллекция '{HISTORY_COLLECTION}' найдена")
    except Exception:
        qdrant.create_collection(
            collection_name=HISTORY_COLLECTION,
            vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
        )
        print(f"✅ Коллекция '{HISTORY_COLLECTION}' создана")

    # 3. Индексы для истории
    indices = [
        ("user_id", models.PayloadSchemaType.KEYWORD),
        ("role", models.PayloadSchemaType.KEYWORD),
        ("message_hash", models.PayloadSchemaType.KEYWORD),
    ]
    for field_name, field_schema in indices:
        try:
            qdrant.create_payload_index(
                collection_name=HISTORY_COLLECTION,
                field_name=field_name,
                field_schema=field_schema,
            )
            print(f"✅ Индекс для '{field_name}' создан")
        except Exception:
            print(f"ℹ️ Индекс для '{field_name}' уже существует")

    # 4. Установка вебхука Telegram
    if TELEGRAM_BOT_TOKEN and WEBHOOK_URL:
        set_webhook_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook?url={WEBHOOK_URL}"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(set_webhook_url)
                print(f"✅ Telegram Webhook установлен: {response.json()}")
            except Exception as e:
                print(f"❌ Ошибка установки Telegram Webhook: {e}")
    else:
        print("⚠️ Переменные TELEGRAM_BOT_TOKEN или WEBHOOK_URL не найдены.")


# ==========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
async def get_embedding(text: str) -> list[float]:
    """Получение эмбеддинга через Jina AI (dim=384)"""
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "jina-embeddings-v3",
        "input": [text],
        "task": "text-matching",
        "dimensions": 384,
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(JINA_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


async def search_knowledge(query: str) -> str:
    try:
        query_vector = await get_embedding(query)
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3,
            with_payload=True,
        )
        if not search_result:
            return ""
        return "\n\n".join([hit.payload.get("text", "") for hit in search_result if hit.payload])
    except Exception as e:
        print(f"⚠️ Ошибка поиска: {e}")
        return ""


async def save_to_history(user_id: str, role: str, content: str):
    """Сохраняет сообщение в историю."""
    try:
        safe_user_id = str(user_id).strip()
        safe_role = str(role).strip()
        safe_content = str(content).strip() if content is not None else ""

        normalized_content = ' '.join(safe_content.split())
        content_key = f"{safe_user_id}_{safe_role}_{normalized_content}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, content_key))

        existing = await asyncio.to_thread(
            qdrant.retrieve,
            collection_name=HISTORY_COLLECTION,
            ids=[point_id],
            with_payload=False,
        )

        if existing:
            print(f"⚠️ Пропускаем дубликат: {normalized_content[:30]}...")
            return

        timestamp = datetime.utcnow().isoformat()
        content_hash = hashlib.md5(content_key.encode('utf-8')).hexdigest()

        await asyncio.to_thread(
            qdrant.upsert,
            collection_name=HISTORY_COLLECTION,
            points=[models.PointStruct(
                id=point_id,
                vector=[1.0],
                payload={
                    "user_id": safe_user_id,
                    "role": safe_role,
                    "content": normalized_content,
                    "message_hash": content_hash,
                    "timestamp": timestamp,
                },
            )],
        )
        print(f"✅ История сохранена: {safe_role} ({len(normalized_content)} симв.)")

    except Exception as e:
        print(f"⚠️ Ошибка сохранения истории: {e}")
        traceback.print_exc()


def get_history(user_id: str, limit: int = 50) -> list[dict]:
    try:
        records, _ = qdrant.scroll(
            collection_name=HISTORY_COLLECTION,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
            limit=limit,
            with_payload=True,
        )
        messages = sorted(
            [r.payload for r in records if r.payload],
            key=lambda x: x.get("timestamp", ""),
        )
        return messages
    except Exception as e:
        print(f"⚠️ Ошибка получения истории: {e}")
        return []


# ==========================================
# 🧠 ЯДРО ЧАТА (для веб-чата)
# ==========================================
async def process_message_core(user_id: str, text: str) -> str:
    if len(text) > 1000:
        return "Сообщение слишком длинное. Максимум 1000 символов."

    if not POLZA_API_KEY:
        return "Ошибка: не настроен ключ Polza AI."

    print(f"🧠 Запрос от {user_id}: '{text[:50]}...'")
    await save_to_history(user_id, "user", text)

    history = get_history(user_id, limit=3)

    chat_history_str = ""
    for msg in history:
        role = msg.get('role', 'unknown')
        role_name = "Пользователь" if role == 'user' else "Ассистент"
        chat_history_str += f"{role_name}: {msg.get('content', '')}\n"

    context = await search_knowledge(text) if JINA_API_KEY else ""
    prompt = ""
    if chat_history_str:
        prompt += f"История диалога:\n{chat_history_str}\n\n"
    if context:
        prompt += f"Контекст:\n{context}\n\n"

    prompt += f"Вопрос: {text}\n\nОтветь кратко, по существу. Максимум 6 предложений."

    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            response = await client.post(
                "https://api.polza.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {POLZA_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "deepseek/deepseek-v4-flash",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 1550,
                },
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"⚠️ Ошибка Polza API: {e}")
            traceback.print_exc()
            return "Извините, произошла ошибка при обращении к ИИ."

    await save_to_history(user_id, "bot", answer)
    return answer


# ==========================================
# 📡 ВЫЗОВ YANDEX CLOUD FUNCTION
# ==========================================
async def call_yandex_function(message_text: str, user_id: str) -> str:
    """
    Отправляет сообщение в Yandex Cloud Function через API Gateway
    и возвращает её ответ.
    """
    payload = {
        "message": message_text,
        "user_id": user_id,
    }
    headers = {
        "Content-Type": "application/json",
    }

    print(f"📡 Отправляем в Yandex: message='{message_text[:80]}...', user_id={user_id}")

    try:
        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(YANDEX_FUNCTION_URL, json=payload, headers=headers) as resp:
                raw = await resp.text()
                print(f"📥 Ответ Yandex (HTTP {resp.status}): {raw[:300]}")

                if resp.status != 200:
                    return f"❌ Yandex вернул ошибку HTTP {resp.status}. Попробуйте позже."

                answer = _extract_answer_from_yandex(raw)
                if not answer:
                    return "❌ Yandex вернул пустой ответ."
                return answer

    except aiohttp.ClientError as e:
        print(f"❌ Сетевая ошибка при вызове Yandex: {e}")
        return "❌ Не удалось связаться с Yandex Cloud."
    except Exception as e:
        print(f"❌ Неизвестная ошибка при вызове Yandex: {e}")
        traceback.print_exc()
        return "❌ Внутренняя ошибка при обращении к Yandex."


def _extract_answer_from_yandex(raw: str) -> str:
    """
    API Gateway + Mangum могут обернуть ответ 1 или 2 раза.
    Разворачиваем рекурсивно, пока не найдём поле с ответом.
    """
    def _dig(obj, depth=0):
        if depth > 5 or obj is None:
            return None
        if isinstance(obj, dict):
            for key in ("response", "answer", "text", "message", "content"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            if "body" in obj:
                body = obj["body"]
                if isinstance(body, str):
                    try:
                        return _dig(json.loads(body), depth + 1)
                    except json.JSONDecodeError:
                        if len(body.strip()) > 5:
                            return body.strip()
                else:
                    return _dig(body, depth + 1)
            if "result" in obj:
                return _dig(obj["result"], depth + 1)
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return _dig(data)


# ==========================================
# 🧵 ФОНОВАЯ ОБРАБОТКА СООБЩЕНИЙ TELEGRAM
# ==========================================
async def process_and_reply(chat_id: int, user_id: str, text: str):
    """
    Фоновая задача:
    1. Извлекает содержимое ссылок (VK, Telegram, обычные сайты) НА RENDER.
    2. Помечает payload маркером [RENDER_LINK], чтобы Yandex-функция
       не уходила в режим накопления длинных текстов.
    3. Удаляет протоколы (https://) из контента, чтобы Yandex не пытался
       парсить ссылку сам (мы это уже сделали).
    4. Передаёт текст + содержимое страницы в Yandex.
    5. Отправляет ответ в Telegram.
    """
    try:
        print(f"🧵 [BG] Начинаем обработку для chat_id={chat_id}")

        # 1. Сообщение «думаю»
        await send_telegram_message(
            chat_id,
            "⏳ Думаю над ответом, это может занять до минуты. Пожалуйста, подождите...",
        )

        # 2. Извлекаем ссылки и парсим их на Render
        urls = extract_urls(text)
        page_text = ""

        if urls:
            page_url = urls[0]
            print(f"🔗 [BG] Найдена ссылка: {page_url}")

            if "t.me/" in page_url or "telegram.me/" in page_url:
                print(f"📱 [BG] Тип: Telegram-канал")
            elif "vk.com/wall" in page_url or "vk.ru/wall" in page_url:
                print(f"📘 [BG] Тип: ВКонтакте")
            else:
                print(f"🌐 [BG] Тип: обычный сайт")

            page_text = await fetch_url_content(page_url)

            if page_text:
                print(f"📄 [BG] Извлечено {len(page_text)} символов из ссылки")
            else:
                print(f"⚠️ [BG] Не удалось извлечь содержимое ссылки {page_url}")

        # 3. Убираем URL из текста пользователя
        text_clean = re.sub(r'https?://\S+', '', text).strip()
        if not text_clean:
            text_clean = "Проанализируй содержимое страницы по ссылке и дай развёрнутый ответ."

        # 4. Формируем payload с маркером [RENDER_LINK]
        if page_text:
            # Убираем протоколы, чтобы Yandex не активировал свою ветку
            # парсинга ссылок (регекс r'https?://[^\s]+' ничего не найдёт)
            page_text_safe = (
                page_text
                .replace("https://", "")
                .replace("http://", "")
            )
            payload_text = (
                f"{RENDER_LINK_MARKER}\n"
                f"{text_clean}\n\n"
                f"Содержимое страницы, которую прислал пользователь:\n"
                f"{page_text_safe}"
            )
        else:
            payload_text = (
                f"{RENDER_LINK_MARKER}\n"
                f"{text_clean}\n\n"
                f"[Содержимое страницы загрузить не удалось. "
                f"Ответь по общим знаниям.]"
            )

        print(f"📤 [BG] Payload для Yandex: {len(payload_text)} симв. "
              f"Начало: {payload_text[:120]!r}")

        # 5. Отправляем в Yandex
        response_text = await call_yandex_function(payload_text, user_id)
        print(f"🧵 [BG] Ответ Yandex: {response_text[:120]}...")

        # 6. Отправляем ответ пользователю
        await send_telegram_message(chat_id, response_text)
        print(f"🧵 [BG] Ответ отправлен в чат {chat_id}")

    except Exception as e:
        print(f"❌ [BG] Ошибка фоновой обработки: {e}")
        traceback.print_exc()
        try:
            await send_telegram_message(chat_id, "Извините, произошла ошибка при обработке.")
        except Exception:
            pass


# ==========================================
# 📱 TELEGRAM ИНТЕГРАЦИЯ
# ==========================================
async def send_telegram_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            result = await resp.json()
            if not result.get("ok"):
                print(f"❌ Ошибка Telegram: {result}")
            else:
                print(f"✅ Отправлено: message_id={result.get('result', {}).get('message_id')}")
            return result


@app.post("/webhook/telegram")
async def telegram_webhook(update: dict):
    if "message" in update:
        message = update["message"]
        chat_id = message["chat"]["id"]
        user_id = f"tg_{str(message['chat']['id'])}"
        if "text" not in message:
            return {"ok": True}
        text = message["text"].strip()

        if text.lower() == "/start":
            await send_telegram_message(chat_id, "Я Феон - верующий ИИ. Чем могу помочь?")
            return {"ok": True}

        asyncio.create_task(process_and_reply(chat_id, user_id, text))
        return {"ok": True}

    return {"ok": True}


# ==========================================
# 🌐 ЭНДПОИНТЫ ДЛЯ ФРОНТЕНДА И АДМИНКИ
# ==========================================
@app.get("/")
def read_root():
    return {"status": "running", "message": "Feon RAG Adapter (TG + Web) работает!"}


@app.post("/add_knowledge")
async def add_knowledge(request: Request):
    try:
        body = await request.json()
        text = body.get("text", "")
        if not text or len(text.strip()) < 10:
            return JSONResponse({"error": "Текст слишком короткий"}, status_code=400)

        doc_vector = await get_embedding(text)
        stable_id = hashlib.md5(text.encode()).hexdigest()

        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[models.PointStruct(
                id=stable_id,
                vector=doc_vector,
                payload={"text": text},
            )],
        )
        return JSONResponse({"status": "success", "message": "Знание добавлено"})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    try:
        import io
        from pypdf import PdfReader
        from docx import Document

        filename = file.filename.lower()
        content = await file.read()
        text = ""

        if filename.endswith('.pdf'):
            reader = PdfReader(io.BytesIO(content))
            text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
        elif filename.endswith('.docx'):
            doc = Document(io.BytesIO(content))
            text = "\n\n".join([para.text for para in doc.paragraphs])
        else:
            return JSONResponse({"error": "Поддерживаются только .pdf и .docx"}, status_code=400)

        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current_chunk) + len(para) <= 800:
                current_chunk += (("\n\n" if current_chunk else "") + para)
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(para) > 800:
                    for i in range(0, len(para), 700):
                        chunks.append(para[i:i + 800])
                current_chunk = ""
        if current_chunk:
            chunks.append(current_chunk)
        chunks = [c for c in chunks if len(c.strip()) > 30]

        success_count = 0
        for i, chunk in enumerate(chunks):
            try:
                doc_vector = await get_embedding(chunk)
                stable_id = hashlib.md5(f"{file.filename}_{i}".encode()).hexdigest()

                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[models.PointStruct(
                        id=stable_id,
                        vector=doc_vector,
                        payload={"text": chunk, "source_file": file.filename},
                    )],
                )
                success_count += 1
            except Exception as e:
                print(f"⚠️ Пропуск фрагмента {i}: {e}")

        return JSONResponse({
            "status": "success",
            "message": f"Добавлено {success_count} из {len(chunks)} фрагментов",
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/query")
async def handle_query(request: Request):
    try:
        body = await request.json()
        message = body.get("message") or body.get("text", "")
        user_id = body.get("user_id", "anonymous")
        if not message:
            return JSONResponse({"error": "Сообщение не может быть пустым"}, status_code=400)

        base_answer = await process_message_core(user_id, message)
        return JSONResponse({"response": base_answer})
    except Exception as e:
        print(f"❌ Ошибка в /query: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/get_history")
async def get_history_endpoint(user_id: str):
    try:
        messages = get_history(user_id, limit=50)
        return JSONResponse({"history": messages})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/get_all_users")
async def get_all_users(request: Request):
    verify_admin(request)
    try:
        records, _ = qdrant.scroll(collection_name=HISTORY_COLLECTION, limit=300, with_payload=True)
        users = {}
        for r in records:
            if r.payload:
                uid = r.payload.get("user_id", "unknown")
                if uid not in users:
                    users[uid] = {
                        "user_id": uid,
                        "message_count": 0,
                        "last_activity": r.payload.get("timestamp", ""),
                    }
                users[uid]["message_count"] += 1
                if r.payload.get("timestamp", "") > users[uid]["last_activity"]:
                    users[uid]["last_activity"] = r.payload.get("timestamp", "")
        sorted_users = sorted(users.values(), key=lambda x: x["last_activity"], reverse=True)
        return JSONResponse({"users": sorted_users, "total": len(sorted_users)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/delete_user")
async def delete_user(user_id: str, request: Request):
    verify_admin(request)
    try:
        qdrant.delete(
            collection_name=HISTORY_COLLECTION,
            points_selector=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
        )
        return JSONResponse({"status": "success", "message": f"Пользователь {user_id} удален"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/get_all_knowledge")
async def get_all_knowledge(request: Request):
    verify_admin(request)
    try:
        records, _ = qdrant.scroll(collection_name=COLLECTION_NAME, limit=500, with_payload=True)
        knowledge_list = []
        for r in records:
            if r.payload:
                knowledge_list.append({
                    "id": r.id,
                    "text": r.payload.get("text", ""),
                    "source_file": r.payload.get("source_file", "Ручной ввод"),
                    "length": len(r.payload.get("text", "")),
                })
        files_stats = {}
        for item in knowledge_list:
            fname = item["source_file"]
            if fname not in files_stats:
                files_stats[fname] = {"name": fname, "chunks": 0, "total_length": 0}
            files_stats[fname]["chunks"] += 1
            files_stats[fname]["total_length"] += item["length"]
        return JSONResponse({
            "knowledge": knowledge_list,
            "total": len(knowledge_list),
            "files": list(files_stats.values()),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/delete_knowledge")
async def delete_knowledge(request: Request):
    verify_admin(request)
    try:
        body = await request.json()
        knowledge_id = body.get("id")
        if not knowledge_id:
            return JSONResponse({"error": "ID не указан"}, status_code=400)
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(points=[knowledge_id]),
        )
        return JSONResponse({"status": "success", "message": f"Знание {knowledge_id} удалено"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/delete_file_knowledge")
async def delete_file_knowledge(file_name: str, request: Request):
    verify_admin(request)
    try:
        records, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="source_file", match=models.MatchValue(value=file_name))]
            ),
            limit=500,
            with_payload=False,
        )
        if not records:
            return JSONResponse({"error": "Файл не найден"}, status_code=404)
        ids_to_delete = [r.id for r in records]
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(points=ids_to_delete),
        )
        return JSONResponse({
            "status": "success",
            "message": f"Удалено {len(ids_to_delete)} фрагментов из файла {file_name}",
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/update_system_prompt")
async def update_system_prompt(request: Request):
    token = request.headers.get("x-admin-token")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Неверный токен администратора")

    try:
        body = await request.json()
        new_prompt = body.get("prompt", "").strip()
        if not new_prompt:
            raise HTTPException(status_code=400, detail="Поле 'prompt' не может быть пустым")

        global SYSTEM_PROMPT
        SYSTEM_PROMPT = new_prompt

        with open("system_prompt.txt", "w", encoding="utf-8") as f:
            f.write(new_prompt)

        return JSONResponse({
            "status": "success",
            "message": "Системный промпт обновлён",
            "new_prompt": new_prompt,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.api_route("/ping", methods=["GET", "HEAD"])
async def ping():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
