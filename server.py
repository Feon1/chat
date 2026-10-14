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
import time


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://feon1.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ
# ==========================================

YANDEX_FUNCTION_URL = os.getenv(
    "YANDEX_FUNCTION_URL",
    "https://d5dq57ou1bsu78horuke.avjje9e3.apigw.yandexcloud.net/query"
)
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")

if not YANDEX_API_KEY:
    print("⚠️ ВНИМАНИЕ: YANDEX_API_KEY не задан.")

BOT_ID = None
BOT_USERNAME = "Feon_ai_bot"
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не задан")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")
POLZA_API_KEY = os.getenv("POLZA_API_KEY")

KNOWLEDGE_MAX_COLLECTION = "xiaozhi_knowledge_max"

POLZA_EMBEDDING_API_KEY = os.getenv("POLZA_EMBEDDING_API_KEY") or POLZA_API_KEY
POLZA_EMBEDDING_URL = os.getenv("POLZA_EMBEDDING_URL", "https://polza.ai/api/v1/embeddings")

COLLECTION_NAME = "xiaozhi_knowledge_max"
HISTORY_COLLECTION = "chat_history_max"

JINA_API_URL = "https://api.jina.ai/v1/embeddings"

LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL")

if not LOCAL_LLM_BASE_URL or not LOCAL_LLM_API_KEY:
    print("⚠️ Локальная модель не настроена: нет LOCAL_LLM_BASE_URL или LOCAL_LLM_API_KEY")
    print("   Команда /model_local будет недоступна")

SETTINGS_COLLECTION = "app_settings"

BUCKET_NAME = os.getenv("BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ru-central1")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "https://storage.yandexcloud.net")

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30,
)
_current_model = "polza"
_SETTINGS_MODEL_ID = "11111111-1111-1111-1111-111111111111"


def get_current_model() -> str:
    return _current_model


async def load_current_model():
    """Читает текущую модель из Qdrant."""
    global _current_model
    try:
        records = await asyncio.to_thread(
            qdrant.retrieve,
            collection_name=SETTINGS_COLLECTION,
            ids=[_SETTINGS_MODEL_ID],
            with_payload=True,
        )
        if records and records[0].payload:
            val = records[0].payload.get("value", "polza")
            if val in ("polza", "local"):
                _current_model = val
                print(f"✅ Модель из Qdrant: {_current_model}")
                return
        print(f"ℹ️ В Qdrant нет настроек, используем 'polza'")
    except Exception as e:
        print(f"ℹ️ Не удалось прочитать настройки из Qdrant: {e}")


async def save_current_model(model: str):
    """Сохраняет выбор модели в Qdrant."""
    global _current_model
    if model not in ("polza", "local"):
        raise ValueError(f"Неизвестная модель: {model}")
    _current_model = model
    try:
        await asyncio.to_thread(
            qdrant.upsert,
            collection_name=SETTINGS_COLLECTION,
            points=[models.PointStruct(
                id=_SETTINGS_MODEL_ID,
                vector=[1.0],
                payload={"key": "current_model", "value": model}
            )]
        )
        print(f"✅ Модель сохранена в Qdrant: {model}")
    except Exception as e:
        print(f"⚠️ Не удалось сохранить модель в Qdrant: {e}")

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


# Загружаем промпт только из переменных окружения
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")

if not SYSTEM_PROMPT or not SYSTEM_PROMPT.strip():
    print("⚠️ SYSTEM_PROMPT не задан в env — использую дефолт")
    SYSTEM_PROMPT = "Ты — Феон, верующий ИИ."
else:
    print(f"✅ SYSTEM_PROMPT из env ({len(SYSTEM_PROMPT)} симв.)")


_ssl_context = ssl.create_default_context(cafile=certifi.where())

RENDER_LINK_MARKER = "[RENDER_LINK]"


def verify_admin(request: Request):
    token = request.headers.get("x-admin-token")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Неверный токен администратора")


# ==========================================
# 🕷️ ИЗВЛЕЧЕНИЕ КОНТЕНТА
# ==========================================

async def fetch_telegram_post(url: str) -> str:
    print(f"📱 [TG] Обрабатываем ссылку: {url}")

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
        print("⚠️ [TG] Приватный канал")
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
                    print(f"⚠️ [TG] HTTP {response.status_code}")
                    continue

                soup = BeautifulSoup(response.text, "lxml")

                target = f"{channel}/{post_id}"
                collected = []

                for post in soup.select(".tgme_widget_message"):
                    data_post = post.get("data-post", "")
                    if data_post == target or data_post.endswith(f"/{post_id}"):
                        txt_el = post.select_one(".tgme_widget_message_text")
                        if txt_el:
                            collected.append(txt_el.get_text(separator="\n", strip=True))
                        cap_el = post.select_one(".tgme_widget_message_caption")
                        if cap_el:
                            collected.append(cap_el.get_text(separator="\n", strip=True))
                        break

                if not collected:
                    for el in soup.select(
                        ".tgme_widget_message_text, .tgme_widget_message_caption"
                    ):
                        txt = el.get_text(separator="\n", strip=True)
                        if txt:
                            collected.append(txt)
                            break

                if not collected:
                    print("⚠️ [TG] Текст поста не найден")
                    continue

                result = "\n\n".join(collected)
                result = re.sub(r'\n{3,}', '\n\n', result).strip()

                MAX_LEN = 15000
                if len(result) > MAX_LEN:
                    result = result[:MAX_LEN] + "...\n[Текст обрезан]"

                print(f"✅ [TG] Извлечено {len(result)} символов")
                return result

            except Exception as e:
                print(f"⚠️ [TG] Ошибка загрузки {web_url}: {e}")
                continue

    print("❌ [TG] Все попытки провалились")
    return ""


async def fetch_vk_post(url: str) -> str:
    match = re.search(r'wall(-?\d+)_(\d+)', url)
    if not match:
        return ""

    owner_id, post_id = match.group(1), match.group(2)
    vk_token = os.getenv("VK_USER_TOKEN") or os.getenv("VK_GROUP_TOKEN")
    if not vk_token:
        return ""

    async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
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
            print(f"⚠️ [VK] {e}")

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
                            return text
        except Exception as e:
            print(f"⚠️ [VK] {e}")

    return ""


async def fetch_url_content(url: str) -> str:
    print(f"🌐 [FETCH] Загружаем: {url}")

    if "t.me/" in url or "telegram.me/" in url:
        tg_text = await fetch_telegram_post(url)
        if tg_text:
            return tg_text
        print("⚠️ [TG] Не удалось")

    if "vk.ru/wall" in url or "vk.com/wall" in url:
        vk_text = await fetch_vk_post(url)
        if vk_text and len(vk_text) > 100:
            return vk_text

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
            print(f"✅ [FETCH] HTTP {response.status_code}")

            soup = BeautifulSoup(response.text, "lxml")

            for element in soup([
                "script", "style", "nav", "footer", "header", "aside", "noscript",
                "button", "form", "iframe", "svg",
            ]):
                element.decompose()

            UI_SELECTORS = [
                ".share", ".sharing", ".social", ".social-share", ".share-buttons",
                "[class*='share']", "[class*='social']",
                "[class*='toolbar']", "[class*='comment']", "[class*='subscribe']",
                "[class*='sidebar']", "[class*='related']", "[class*='recommend']",
                "[class*='cookie']", "[class*='banner']", "[class*='advert']",
            ]
            for selector in UI_SELECTORS:
                for el in soup.select(selector):
                    el.decompose()

            text = soup.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            cleaned = "\n".join(lines)

            cleaned = "\n".join(
                line for line in cleaned.splitlines()
                if len(line.strip()) > 3
            )

            NOISE_LINES = {
                "поделиться", "вернуться к странице", "показать список",
                "посты сообщества", "читать далее", "комментировать",
                "подписаться", "репост",
            }
            cleaned = "\n".join(
                line for line in cleaned.splitlines()
                if line.strip().lower() not in NOISE_LINES
            )

            if len(cleaned) < 300:
                print(f"⚠️ [FETCH] Слишком короткий ({len(cleaned)})")
                return ""

            letters = sum(c.isalpha() for c in cleaned)
            if letters / len(cleaned) < 0.5:
                print(f"⚠️ [FETCH] Мало букв")
                return ""

            MAX_LEN = 15000
            if len(cleaned) > MAX_LEN:
                cleaned = cleaned[:MAX_LEN] + "...\n[Текст обрезан]"

            print(f"✅ [FETCH] Извлечено {len(cleaned)} символов")
            return cleaned

    except Exception as e:
        print(f"❌ [FETCH] Ошибка загрузки {url}: {e}")
        return ""


def extract_urls(text: str) -> list[str]:
    return re.findall(r'https?://[^\s]+', text)


def extract_text_from_rich_message(rich: dict) -> str:
    parts = []

    def walk(node):
        if isinstance(node, dict):
            for key in ("text", "content", "value"):
                val = node.get(key)
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
            for v in node.values():
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(rich)
    cleaned = []
    for p in parts:
        if not cleaned or cleaned[-1] != p:
            cleaned.append(p)
    return "\n\n".join(cleaned)


# ==========================================
# ИНИЦИАЛИЗАЦИЯ ПРИ СТАРТЕ
# ==========================================
@app.on_event("startup")
async def startup_event():
    # 1. Коллекция для базы знаний (Polza dim=1536)
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        size = info.config.params.vectors.size
        print(f"✅ Коллекция '{COLLECTION_NAME}' найдена (dim={size})")
        if size != 1536:
            print(f"⚠️ ВНИМАНИЕ: размерность {size} не совпадает с ожидаемой 1536!")
    except Exception:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        )
        print(f"✅ Коллекция '{COLLECTION_NAME}' создана (dim=1536)")

    # 2. Коллекция для истории
    try:
        qdrant.get_collection(HISTORY_COLLECTION)
        print(f"✅ Коллекция '{HISTORY_COLLECTION}' найдена")
    except Exception:
        qdrant.create_collection(
            collection_name=HISTORY_COLLECTION,
            vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
        )
        print(f"✅ Коллекция '{HISTORY_COLLECTION}' создана")

    # 2.5. Коллекция для настроек
    try:
        qdrant.get_collection(SETTINGS_COLLECTION)
        print(f"✅ Коллекция '{SETTINGS_COLLECTION}' найдена")
    except Exception:
        qdrant.create_collection(
            collection_name=SETTINGS_COLLECTION,
            vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
        )
        print(f"✅ Коллекция '{SETTINGS_COLLECTION}' создана")

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
            print(f"✅ Индекс '{field_name}' создан")
        except Exception:
            print(f"ℹ️ Индекс '{field_name}' уже есть")

    # 4. Получаем bot_id + webhook
    global BOT_ID, BOT_USERNAME
    if TELEGRAM_BOT_TOKEN:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
                )
                data = r.json()
                BOT_ID = data.get("result", {}).get("id")
                BOT_USERNAME = data.get("result", {}).get("username", "Feon_ai_bot")
                print(f"✅ Bot ID: {BOT_ID}, username: @{BOT_USERNAME}")
        except Exception as e:
            print(f"⚠️ Не удалось получить bot_id: {e}")
            BOT_ID = None

    if TELEGRAM_BOT_TOKEN and WEBHOOK_URL:
        import json as _json
        allowed = ["message", "channel_post", "edited_message", "edited_channel_post"]

        set_webhook_url = (
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook"
            f"?url={WEBHOOK_URL}"
            f"&allowed_updates={_json.dumps(allowed)}"
            f"&drop_pending_updates=true"
        )

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(set_webhook_url)
                print(f"✅ Telegram Webhook установлен: {response.json()}")
            except Exception as e:
                print(f"❌ Ошибка установки Webhook: {e}")
    else:
        print("⚠️ TELEGRAM_BOT_TOKEN или WEBHOOK_URL не найдены")

    # 5. Регистрация команд и кнопки меню
    if TELEGRAM_BOT_TOKEN:
        commands = [
            {"command": "start", "description": "Начать диалог с Феоном"},
            {"command": "help",  "description": "Справка и примеры"},
            {"command": "menu",  "description": "Меню быстрых действий"},
            {"command": "about", "description": "О Феоне"},
            {"command": "model", "description": "Показать/сменить модель"},
        ]
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setMyCommands",
                    json={"commands": commands}
                )
                print(f"✅ Команды зарегистрированы: {r.json().get('ok')}")

                r2 = await client.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setChatMenuButton",
                    json={"menu_button": {"type": "commands"}}
                )
                print(f"✅ Menu button установлен: {r2.json().get('ok')}")
        except Exception as e:
            print(f"⚠️ Не удалось установить меню: {e}")

    # 6. Загружаем сохранённую модель из Qdrant
    await load_current_model()


# ==========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

async def get_embedding(text: str) -> list[float]:
    """Эмбеддинг через Polza (dim=1536) — для xiaozhi_knowledge_max."""
    if not POLZA_EMBEDDING_API_KEY:
        raise ValueError("POLZA_EMBEDDING_API_KEY не задан")
    headers = {"Authorization": f"Bearer {POLZA_EMBEDDING_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "text-embedding-3-small", "input": [text]}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(POLZA_EMBEDDING_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


async def search_knowledge(query: str) -> str:
    """RAG-поиск в xiaozhi_knowledge_max (dim=1536, Polza)."""
    if not POLZA_EMBEDDING_API_KEY:
        print("⚠️ [SEARCH] Нет POLZA_EMBEDDING_API_KEY")
        return ""

    try:
        query_vector = await get_embedding(query)

        if hasattr(qdrant, "query_points"):
            result = await asyncio.to_thread(
                qdrant.query_points,
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=5,
                with_payload=True,
                score_threshold=0.45,
            )
            search_result = result.points if hasattr(result, "points") else []
        else:
            search_result = await asyncio.to_thread(
                qdrant.search,
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=5,
                with_payload=True,
                score_threshold=0.45,
            )

        if not search_result:
            print("⏭️ [SEARCH] Ничего не найдено")
            return ""

        print(f"✅ [SEARCH] Найдено {len(search_result)} чанков "
              f"(score: {search_result[0].score:.2f}..{search_result[-1].score:.2f})")

        return "\n\n".join([hit.payload.get("text", "") for hit in search_result if hit.payload])
    except Exception as e:
        print(f"⚠️ Ошибка поиска: {e}")
        traceback.print_exc()
        return ""


async def save_to_history(user_id: str, role: str, content: str):
    try:
        safe_user_id = str(user_id).strip()
        safe_role = str(role).strip()
        safe_content = str(content).strip() if content is not None else ""

        safe_content = safe_content.replace(
            "[Содержимое страницы загрузить не удалось. Ответь по общим знаниям.]", ""
        ).strip()
        safe_content = safe_content.replace(
            "Ознакомься с содержимым по ссылке и дай развёрнутый ответ.", ""
        ).strip()

        if not safe_content:
            return

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


async def get_history(user_id: str, limit: int = 50) -> list[dict]:
    """Асинхронная обёртка над Qdrant scroll — не блокирует event loop."""
    try:
        records, _ = await asyncio.to_thread(
            qdrant.scroll,
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

    history = await get_history(user_id, limit=3)

    chat_history_str = ""
    for msg in history:
        role = msg.get('role', 'unknown')
        role_name = "Пользователь" if role == 'user' else "Ассистент"
        chat_history_str += f"{role_name}: {msg.get('content', '')}\n"

    context = await search_knowledge(text) if POLZA_EMBEDDING_API_KEY else ""

    # ---------- Формирование промпта для веб-чата ----------
    print(f"📄 [CONTEXT] Итоговый контекст (первые 300 символов): "
          f"{context[:300] if context else 'пусто'}")

    COMMON_TAIL = (
        "Сначала кратко откликнись на текущее сообщение пользователя (1–2 абзаца, по существу). "
        "Затем плавно перейди к общим философским и богословским размышлениям на эту тему, "
        "опираясь на идеи из контекста и базы знаний (2–3 абзаца). "
        "Не используй слова 'часть', 'раздел', 'во-первых' и т.п. "
        "Пусть переход будет естественным — как в разговоре мудрого собеседника. "
        "Не повторяй формулировки из своих прошлых ответов. "
        "Ответ должен быть содержательным, но не длиннее 1500 символов."
    )

    # Проверка: вопрос ли это (по знаку "?" или ключевым словам как отдельных слов)
    _q_re = re.compile(r"\b(как|почему|что|зачем|когда|где|кто)\b", re.IGNORECASE)
    is_question = ("?" in text) or bool(_q_re.search(text))

    if not is_question:
        # Ветка: обычный текст без явного вопроса
        prompt = (
            f"[Источник: web]\n\n"
            f"История:\n{chat_history_str}\n\n"
            f"Пользователь прислал текст:\n{text}\n\n"
            f"Дай связный, естественный ответ, как в личной беседе. "
            f"Не используй нумерацию, заголовки, списки, маркеры. "
            f"Не перечисляй пункты — излагай мысли плавно, абзацами. "
            f"Опирайся на философские и богословские идеи из системного промпта, "
            f"контекста и базы знаний, не цитируй их дословно. "
            f"Укажи пользователю путь Богу на основе базы знаний. "
            f"Если пользователь прислал ссылку, проанализируй содержимое страницы "
            f"(оно есть в контексте) и дай ответ. "
            f"Оформи ответ в одном сообщении, без заголовков и разделителей. "
            f"{COMMON_TAIL}"
        )
    else:
        # Ветка: есть вопрос
        prompt = (
            f"[Источник: web]\n\n"
            f"История:\n{chat_history_str}\n\n"
            f"Вопрос пользователя: {text}\n\n"
            f"Дай прямой, связный ответ, без нумерации, списков и заголовков. "
            f"Излагай мысли как в разговоре — плавно и цельно. "
            f"Используй философские и богословские идеи из системного промпта, "
            f"базы знаний и контекста для обоснования, но не копируй их структуру. "
            f"Укажи пользователю путь Богу на основе базы знаний. "
            f"Если пользователь прислал ссылку, проанализируй содержимое страницы "
            f"(оно есть в контексте) и дай ответ. "
            f"Оформи ответ в одном сообщении, без заголовков и разделителей. "
            f"{COMMON_TAIL}"
        )

    # Добавление контекста из базы знаний
    if context:
        context_intro = (
            "Ниже приведены философские и богословские фрагменты "
            "из базы знаний, релевантные запросу:\n\n"
        )
        prompt += f"\n\n{context_intro}{context}\n\n"
        prompt += (
            "Проанализируй текст пользователя через призму этих фрагментов. "
            "Используй их для аргументации, сравнения и глубокого осмысления. "
            "Не просто цитируй — интерпретируй. "
            "Если пользователь прислал ссылку, проанализируй содержимое страницы "
            "(оно есть в контексте) и дай ответ. "
            "Укажи пользователю путь Богу на основе базы знаний. "
            "Оформи ответ в одном сообщении, без заголовков и разделителей. "
            "Сначала кратко откликнись на текущее сообщение пользователя (1–2 абзаца, по существу). "
            "Затем плавно перейди к общим философским и богословским размышлениям на эту тему, "
            "опираясь на идеи из контекста и базы знаний (2–3 абзаца). "
            "Не используй слова 'часть', 'раздел', 'во-первых' и т.п. "
            "Пусть переход будет естественным — как в разговоре мудрого собеседника. "
            "Не повторяй формулировки из своих прошлых ответов. "
            "Ответ должен быть содержательным, но не длиннее 1500 символов."
        )

    print(f"📝 [PROMPT] Полная длина: {len(prompt)} символов")
    print(f"📝 [PROMPT] Первые 600 символов:\n{prompt[:600]}...")

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
                    "temperature": 0.7,
                    "max_tokens": 2500,
                    "thinking": {"type": "disabled"},
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
    payload = {
        "message": message_text,
        "user_id": user_id,
        "platform": "Telegram",
    }
    headers = {
        "Content-Type": "application/json",
    }

    print(f"📡 Отправляем в Yandex: message='{message_text[:80]}...', user_id={user_id}")

    try:
        timeout = aiohttp.ClientTimeout(total=10, connect=5, sock_connect=5, sock_read=8)
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
# 🚑 ЛОКАЛЬНЫЙ FALLBACK
# ==========================================

async def get_polza_embedding(text: str) -> list[float]:
    if not POLZA_EMBEDDING_API_KEY:
        raise ValueError("POLZA_EMBEDDING_API_KEY не задан")

    headers = {
        "Authorization": f"Bearer {POLZA_EMBEDDING_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {"model": "text-embedding-3-small", "input": [text]}

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(POLZA_EMBEDDING_URL, headers=headers, json=data)
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]


async def search_knowledge_max(query: str) -> str:
    if not POLZA_EMBEDDING_API_KEY:
        print("⚠️ [FALLBACK-RAG] Нет POLZA_EMBEDDING_API_KEY")
        return ""

    try:
        vec = await get_polza_embedding(query)
    except Exception as e:
        print(f"⚠️ [FALLBACK-RAG] Эмбеддинг упал: {type(e).__name__}: {e}")
        return ""

    points = []
    try:
        if hasattr(qdrant, "query_points"):
            result = await asyncio.to_thread(
                qdrant.query_points,
                collection_name=KNOWLEDGE_MAX_COLLECTION,
                query=vec,
                limit=5,
                with_payload=True,
                score_threshold=0.45,
            )
            points = result.points if hasattr(result, "points") else []
        else:
            points = await asyncio.to_thread(
                qdrant.search,
                collection_name=KNOWLEDGE_MAX_COLLECTION,
                query_vector=vec,
                limit=5,
                with_payload=True,
                score_threshold=0.45,
            )
    except Exception as e:
        print(f"⚠️ [FALLBACK-RAG] Qdrant упал: {type(e).__name__}: {e}")
        traceback.print_exc()
        return ""

    if not points:
        print("⏭️ [FALLBACK-RAG] Ничего не найдено")
        return ""

    print(f"✅ [FALLBACK-RAG] Найдено {len(points)} чанков "
          f"(score: {points[0].score:.2f}..{points[-1].score:.2f})")

    result = "\n\n".join([
        p.payload.get("text", "") for p in points if p.payload
    ])

# Ограничим суммарный размер контекста, чтобы не раздувать промпт
    MAX_CONTEXT_CHARS = 8000
    if len(result) > MAX_CONTEXT_CHARS:
        result = result[:MAX_CONTEXT_CHARS] + "\n\n[Контекст сокращён]"
        print(f"✂️ [FALLBACK-RAG] Контекст обрезан до {MAX_CONTEXT_CHARS} символов")

    return result


async def call_local_llm(messages: list, temperature: float = 0.7, max_tokens: int = 500) -> str:
    """Вызов локальной модели через Tailscale (OpenAI-совместимый API)."""
    print(f"🖥️ [LOCAL-LLM] Отправка на {LOCAL_LLM_BASE_URL}, model={LOCAL_LLM_MODEL}")

    # Tailscale Funnel медленно отвечает на TLS handshake — нужен запас
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(180.0, connect=60.0, write=30.0, pool=30.0),
    ) as client:
        try:
            r = await client.post(
                f"{LOCAL_LLM_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {LOCAL_LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LOCAL_LLM_MODEL,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            r.raise_for_status()
            data = r.json()
            if data.get("choices"):
                content = data["choices"][0].get("message", {}).get("content") or ""
                print(f"✅ [LOCAL-LLM] Ответ: {len(content)} символов")
                return content
            print(f"⚠️ [LOCAL-LLM] Нет choices: {json.dumps(data, ensure_ascii=False)[:300]}")
            return ""
        except Exception as e:
            print(f"❌ [LOCAL-LLM] Ошибка: {type(e).__name__}: {e}")
            traceback.print_exc()
            return ""


async def call_polza_llm(messages: list, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """Вызов Polza (DeepSeek V4 Flash)."""
    print(f"☁️ [POLZA-LLM] Отправка в Polza, model=deepseek-v4-flash")

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0)) as client:
        try:
            polza_json = {
                "model": "deepseek/deepseek-v4-flash",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "thinking": {"type": "disabled"},
            }
            r = await client.post(
                "https://api.polza.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {POLZA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=polza_json,
            )
            r.raise_for_status()
            data = r.json()

            if "usage" in data:
                print(f"🔍 [POLZA-LLM] usage={data['usage']}")

            if data.get("choices"):
                ch = data["choices"][0]
                msg = ch.get("message", {}) or {}
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning") or ""
                print(f"🔍 [POLZA-LLM] finish={ch.get('finish_reason')}, "
                      f"content_len={len(content)}, reasoning_len={len(reasoning)}")
                return content
            print(f"⚠️ [POLZA-LLM] Нет choices")
            return ""
        except Exception as e:
            print(f"❌ [POLZA-LLM] Ошибка: {type(e).__name__}: {e}")
            traceback.print_exc()
            return ""


async def call_llm(messages: list, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """Роутер: вызывает модель согласно текущей настройке."""
    model = get_current_model()
    print(f"🎛️ [LLM] Используем модель: {model}")
    if model == "local":
        answer = await call_local_llm(messages, temperature, max_tokens)
        if not answer or not answer.strip():
            print(f"⚠️ [LLM] Локальная не ответила — фолбэк на Polza")
            answer = await call_polza_llm(messages, temperature, max_tokens)
        return answer
    else:
        return await call_polza_llm(messages, temperature, max_tokens)


async def local_fallback(payload_text: str, user_id: str,
                         platform: str = "Telegram",
                         is_channel: bool = False) -> str:
    print("🚑 [FALLBACK] Yandex недоступен — генерирую ответ локально")

    # 1. Срезаем маркер
    text = payload_text
    if text.startswith(RENDER_LINK_MARKER):
        text = text[len(RENDER_LINK_MARKER):].lstrip()

    # 2. Убираем заглушки
    text = text.replace(
        "[Содержимое страницы загрузить не удалось. Ответь по общим знаниям.]", ""
    ).strip()
    text = text.replace(
        "Ознакомься с содержимым по ссылке и дай развёрнутый ответ.", ""
    ).strip()

    if not text:
        return "Извините, я не смог получить вопрос."

    # 3. RAG
    from_current = get_current_model()
    if from_current == "local":
        print("🖥️ [FALLBACK] Локальная модель — RAG урезан до 1-2 чанков")
        context = await search_knowledge(text)
        if context:
            context = context[:1500]
            print(f"📄 [FALLBACK] RAG-контекст (сокращён): {len(context)} символов")
        else:
            print("⏭️ [FALLBACK] RAG пуст")
    else:
        context = await search_knowledge_max(text)
        if context:
            print(f"📄 [FALLBACK] RAG-контекст: {len(context)} символов")
        else:
            print("⏭️ [FALLBACK] RAG пуст")

    # 4. История
    history = await get_history(user_id, limit=3)
    chat_history_str = "\n".join([
        f"{'Пользователь' if m.get('role')=='user' else 'Ассистент'}: {m.get('content', '')}"
        for m in history if m.get("content")
    ])

    # 5. Формирование промпта
    current_model = get_current_model()

    # Отладочный вывод контекста
    print(f"📄 [CONTEXT] Итоговый контекст (первые 300 символов): "
          f"{context[:300] if context else 'пусто'}")

    # --- Для локальной модели оставляем минимальный промпт (Qwen на телефоне) ---
    if current_model == "local":
        prompt = f"Вопрос: {text}\n\n"
        if context:
            prompt += f"Контекст (кратко):\n{context[:1200]}\n\n"
        prompt += "Ответь коротко, 3-5 предложений, по существу."

    # --- Для Polza — полноценный промпт с ветвлением ---
    else:
        is_channel_post = bool(is_channel)

        COMMON_TAIL = (
            "Сначала кратко откликнись на текущее сообщение пользователя "
            "(1–2 абзаца, по существу). "
            "Затем плавно перейди к общим философским и богословским размышлениям "
            "на эту тему, опираясь на идеи из контекста и базы знаний (2–3 абзаца). "
            "Не используй слова 'часть', 'раздел', 'во-первых' и т.п. "
            "Пусть переход будет естественным — как в разговоре мудрого собеседника. "
            "Не повторяй формулировки из своих прошлых ответов. "
            "Ответ должен быть содержательным, но не длиннее 1500 символов."
        )

        # --- Ветка 1: пост из канала ---
        if is_channel_post:
            prompt = (
                f"[Источник: {platform}]\n\n"
                f"=== СООБЩЕНИЕ ПОСТА ===\n{text}\n=== КОНЕЦ СООБЩЕНИЯ ===\n\n"
                f"Твоя задача:\n"
                f"1) СНАЧАЛА кратко перескажи суть этого поста — 1–2 предложения, "
                f"по существу, без воды и без оценочных суждений.\n"
                f"2) ЗАТЕМ дай свой отклик на этот пост — 2–3 абзаца живым языком, "
                f"опираясь на философские и богословские идеи из системного промпта "
                f"и базы знаний. "
                f"Не используй нумерацию, списки, заголовки. "
                f"Не начинай отклик со слов «Знаешь», «Видишь ли», «Дело в том, что».\n"
                f"3) Укажи пользователю путь Богу на основе базы знаний — но ненавязчиво, "
                f"вплетая в текст, а не отдельной строкой.\n\n"
                f"Формат ответа:\n"
                f"Кратко о сути вопроса: <1–2 предложения пересказа>\n\n"
                f"<2–3 абзаца отклика>\n\n"
                f"Ответ — не длиннее 1500 символов."
            )

        # --- Ветка 2: обычный текст без явного вопроса ---
        else:
            _q_re = re.compile(r"\b(как|почему|что|зачем|когда|где|кто)\b", re.IGNORECASE)
            is_question = ("?" in text) or bool(_q_re.search(text))

            if not is_question:
                prompt = (
                    f"[Источник: {platform}]\n\n"
                    f"История:\n{chat_history_str}\n\n"
                    f"Пользователь прислал текст:\n{text}\n\n"
                    f"Дай связный, естественный ответ, как в личной беседе. "
                    f"Не используй нумерацию, заголовки, списки, маркеры. "
                    f"Не перечисляй пункты — излагай мысли плавно, абзацами. "
                    f"Опирайся на философские и богословские идеи из системного промпта, "
                    f"контекста и базы знаний, не цитируй их дословно. "
                    f"Укажи пользователю путь Богу на основе базы знаний. "
                    f"Если пользователь прислал ссылку, проанализируй содержимое страницы "
                    f"(оно есть в контексте) и дай ответ. "
                    f"Оформи ответ в одном сообщении, без заголовков и разделителей. "
                    f"{COMMON_TAIL}"
                )
            else:
                prompt = (
                    f"[Источник: {platform}]\n\n"
                    f"История:\n{chat_history_str}\n\n"
                    f"Вопрос пользователя: {text}\n\n"
                    f"Дай прямой, связный ответ, без нумерации, списков и заголовков. "
                    f"Излагай мысли как в разговоре — плавно и цельно. "
                    f"Используй философские и богословские идеи из системного промпта, "
                    f"базы знаний и контекста для обоснования, но не копируй их структуру. "
                    f"Укажи пользователю путь Богу на основе базы знаний. "
                    f"Если пользователь прислал ссылку, проанализируй содержимое страницы "
                    f"(оно есть в контексте) и дай ответ. "
                    f"Оформи ответ в одном сообщении, без заголовков и разделителей. "
                    f"{COMMON_TAIL}"
                )

        # --- Добавление контекста из базы знаний ---
        if context:
            if is_channel_post:
                # Для постов из канала — просто подмешиваем справочный материал
                prompt += (
                    f"\n\nСправочный материал из базы знаний "
                    f"(используй по необходимости):\n{context}"
                )
            else:
                context_intro = (
                    "Ниже приведены философские и богословские фрагменты "
                    "из базы знаний, релевантные запросу:\n\n"
                )
                prompt += f"\n\n{context_intro}{context}\n\n"
                prompt += (
                    "Проанализируй текст пользователя через призму этих фрагментов. "
                    "Используй их для аргументации, сравнения и глубокого осмысления. "
                    "Не просто цитируй — интерпретируй. "
                    "Если пользователь прислал ссылку, проанализируй содержимое страницы "
                    "(оно есть в контексте) и дай ответ. "
                    "Укажи пользователю путь Богу на основе базы знаний. "
                    "Оформи ответ в одном сообщении, без заголовков и разделителей. "
                    "Сначала кратко откликнись на текущее сообщение пользователя "
                    "(1–2 абзаца, по существу). "
                    "Затем плавно перейди к общим философским и богословским размышлениям "
                    "на эту тему, опираясь на идеи из контекста и базы знаний (2–3 абзаца). "
                    "Не используй слова 'часть', 'раздел', 'во-первых' и т.п. "
                    "Пусть переход будет естественным — как в разговоре мудрого собеседника. "
                    "Не повторяй формулировки из своих прошлых ответов. "
                    "Ответ должен быть содержательным, но не длиннее 1500 символов."
                )

    print(f"📝 [PROMPT] Полная длина: {len(prompt)} символов")
    print(f"📝 [PROMPT] Первые 600 символов:\n{prompt[:600]}...")

    # 6. Вызов LLM (Polza или локальная — по текущей настройке)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    answer = await call_llm(messages, temperature=0.7, max_tokens=2000)
    print(f"✅ [FALLBACK] Ответ получен: {len(answer) if answer else 0} символов")

    # 7. Сохраняем в историю
    try:
        await save_to_history(user_id, "bot", answer)
    except Exception as e:
        print(f"⚠️ [FALLBACK] Не удалось сохранить в историю: {e}")

    return answer


# ==========================================
# 🧵 ФОНОВАЯ ОБРАБОТКА СООБЩЕНИЙ TELEGRAM
# ==========================================
async def process_and_reply(chat_id: int, user_id: str, text: str,
                            is_channel: bool = False, reply_to_message_id: int = None,
                            post_link: str = None, is_group: bool = False):
    try:
        print(f"🧵 [BG] Начинаем обработку для chat_id={chat_id}, "
              f"reply_to={reply_to_message_id}")

        if not is_channel and not is_group:
            await send_telegram_message(
                chat_id,
                "⏳ Думаю над ответом, это может занять до минуты. Пожалуйста, подождите...",
            )

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
                print(f"⚠️ [BG] Не удалось извлечь содержимое {page_url}")

        text_clean = re.sub(r'https?://\S+', '', text).strip()
        if not text_clean and urls:
            text_clean = "Проанализируй содержимое страницы по ссылке и дай развёрнутый ответ."

        if urls and not page_text:
            print(f"⚠️ [BG] Контент по ссылке не извлечён")
            await send_telegram_message(
                chat_id,
                "⚠️ Не удалось загрузить содержимое по ссылке "
                "(приватный канал, удалённый пост, пейволл или сайт с защитой). "
                "Отвечу по общим знаниям."
            )

        if not urls:
            if len(text) > 1900:
                payload_text = f"{RENDER_LINK_MARKER}\n{text}"
                print(f"📤 [BG] Длинный текст без ссылки — с маркером ({len(payload_text)} симв.)")
            else:
                payload_text = text
                print(f"📤 [BG] Без ссылки — простой текст ({len(payload_text)} симв.)")

        elif page_text:
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

        # 🚑 FALLBACK
        is_yandex_failed = (
            not response_text
            or response_text.startswith("❌")
            or response_text.startswith("Извините, произошла ошибка")
            or response_text.startswith("Извините, я не смог сформулировать")
        )
        if is_yandex_failed:
            print("🚑 [BG] Yandex недоступен — переключаюсь на локальный fallback")
            try:
                response_text = await local_fallback(
                    payload_text, user_id,
                    platform="Telegram",
                    is_channel=is_channel,
                )
            except Exception as e:
                print(f"❌ [BG] Fallback тоже упал: {e}")
                traceback.print_exc()
                response_text = "Извините, сервис временно недоступен. Попробуйте позже."

       
        # Если ответ пустой — не отправляем в канал вообще
        if not response_text or not response_text.strip():
            print("⚠️ [BG] Пустой ответ — в канал не отправляем")
            return

        if is_channel and post_link:
            final_text = (
                f"@Feon_ai_bot говорит:\n\n"
                f"{response_text}\n\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🔗 Ссылка на пост: {post_link}"
            )
        else:
            final_text = response_text
        

        await send_telegram_message(
            chat_id,
            final_text,
            reply_to_message_id=reply_to_message_id if is_channel else None,
        )
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
async def send_telegram_message(chat_id, text, reply_to_message_id=None, parse_mode=None):
    MAX_LEN = 4000
    parts = []
    if len(text) <= MAX_LEN:
        parts = [text]
    else:
        paragraphs = text.split("\n\n")
        current = ""
        for p in paragraphs:
            if len(current) + len(p) + 2 <= MAX_LEN:
                current += (("\n\n" if current else "") + p)
            else:
                if current:
                    parts.append(current)
                while len(p) > MAX_LEN:
                    parts.append(p[:MAX_LEN])
                    p = p[MAX_LEN:]
                current = p
        if current:
            parts.append(current)

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    async with aiohttp.ClientSession() as session:
        for i, part in enumerate(parts, 1):
            prefix = f"[{i}/{len(parts)}] " if len(parts) > 1 else ""
            payload = {"chat_id": chat_id, "text": prefix + part}

            if reply_to_message_id:
                payload["reply_to_message_id"] = reply_to_message_id
                payload["allow_sending_without_reply"] = True

            if parse_mode:
                payload["parse_mode"] = parse_mode

            async with session.post(url, json=payload) as resp:
                result = await resp.json()
                if not result.get("ok"):
                    print(f"❌ Ошибка Telegram (часть {i}): {result}")
                else:
                    print(f"✅ Отправлено: message_id={result.get('result', {}).get('message_id')}")

            reply_to_message_id = None
    return True


# ==========================================
# 🎯 WEBHOOK
# ==========================================
@app.post("/webhook/telegram")
async def telegram_webhook(update: dict):
    import json as _json
    update_types = [k for k in update.keys() if k != "update_id"]
    print(f"📩 [RAW] types={update_types}")

    # ============ ПОСТЫ В КАНАЛЕ ============
    if "channel_post" in update:
        post = update["channel_post"]
        from_info = post.get("from", {}) or {}

        if from_info.get("is_bot") and from_info.get("username") not in ("Channel_Bot",):
            print("⏭️ [CHANNEL] Пост от бота — игнорируем")
            return {"ok": True}

        chat_id = post["chat"]["id"]
        chat_title = post["chat"].get("title", "канал")
        user_id = f"tg_channel_{chat_id}"

        text = (post.get("text") or post.get("caption") or "").strip()

        if not text and "rich_message" in post:
            print("💎 [CHANNEL] Обнаружен rich_message")
            text = extract_text_from_rich_message(post["rich_message"])
            print(f"💎 [CHANNEL] Извлечено {len(text)} символов")

        if not text:
            keys = [k for k in post.keys() if k != "chat"]
            print(f"🔍 [CHANNEL-RAW] Пост без текста. Ключи: {keys}")
            return {"ok": True}

        if len(text) < 15:
            print(f"⏭️ [CHANNEL] Пост слишком короткий")
            return {"ok": True}

        for t in ["#феон", "#feon", f"@{BOT_USERNAME}"]:
            text = re.sub(re.escape(t), "", text, flags=re.IGNORECASE).strip()

        post_message_id = post["message_id"]

        forward_chat = post.get("forward_from_chat") or {}
        forward_msg_id = post.get("forward_from_message_id")

        if forward_chat and forward_chat.get("type") == "channel" and forward_msg_id:
            source_username = forward_chat.get("username")
            source_id = forward_chat.get("id")
            source_title = forward_chat.get("title", "канал-источник")

            if source_username:
                post_link = f"https://t.me/{source_username}/{forward_msg_id}"
            else:
                internal = str(source_id).replace("-100", "")
                post_link = f"https://t.me/c/{internal}/{forward_msg_id}"

            print(f"📣 [CHANNEL] Пересылка из «{source_title}» (msg_id={forward_msg_id})")
            print(f"🔗 [CHANNEL] Ссылка на ИСХОДНЫЙ пост: {post_link}")
        else:
            channel_username = post["chat"].get("username")
            if channel_username:
                post_link = f"https://t.me/{channel_username}/{post_message_id}"
            else:
                internal_id = str(chat_id).replace("-100", "")
                post_link = f"https://t.me/c/{internal_id}/{post_message_id}"

            print(f"📣 [CHANNEL] Своя публикация (msg_id={post_message_id})")

        print(f"📝 Текст: {text[:80]}...")

        asyncio.create_task(process_and_reply(
            chat_id, user_id, text,
            is_channel=True,
            reply_to_message_id=post_message_id,
            post_link=post_link,
        ))
        return {"ok": True}

    # ============ ОБЫЧНЫЕ СООБЩЕНИЯ ============
    if "message" in update:
        message = update["message"]
        from_info = message.get("from", {}) or {}
        sender_chat = message.get("sender_chat", {}) or {}

        # 1. Своё сообщение бота
        if BOT_ID and from_info.get("id") == BOT_ID:
            print(f"⏭️ [MESSAGE] Своё сообщение — игнорируем")
            return {"ok": True}

        # 2. Сообщение от лица канала
        if sender_chat and sender_chat.get("type") == "channel":
            print(f"⏭️ [MESSAGE] Сообщение от канала — игнорируем")
            return {"ok": True}

        # 3. Настоящие боты
        is_real_bot = (
            from_info.get("is_bot")
            and from_info.get("username") not in ("GroupAnonymousBot", "Channel_Bot")
        )
        if is_real_bot:
            print(f"⏭️ [MESSAGE] Сообщение от бота — игнорируем")
            return {"ok": True}

        chat_id = message["chat"]["id"]
        chat_type = message["chat"].get("type", "private")

        text = (message.get("text") or message.get("caption") or "").strip()
        if not text and "rich_message" in message:
            text = extract_text_from_rich_message(message["rich_message"])
            print(f"💎 [MESSAGE] Из rich_message извлечено {len(text)} символов")
        if not text:
            return {"ok": True}

        if text.startswith("⏳") or text.startswith("❌") or text.startswith("⚠️") or text.startswith("@Feon_ai_bot говорит:"):
            print("⏭️ [MESSAGE] Служебное — игнорируем")
            return {"ok": True}

        if chat_type in ("group", "supergroup"):
            if sender_chat and sender_chat.get("id"):
                user_id = f"tg_anon_{sender_chat['id']}_{chat_id}"
            elif from_info.get("id"):
                user_id = f"tg_user_{from_info['id']}"
            else:
                user_id = f"tg_{chat_id}"
        else:
            user_id = f"tg_{from_info.get('id') or chat_id}"

        # ФИЛЬТР ГРУПП
        if chat_type in ("group", "supergroup"):
            mention_triggers = [
                f"@{BOT_USERNAME}",
                f"#{BOT_USERNAME}",
                "#феон", "#feon", "#Феон", "#Feon",
            ]
            text_lower = text.lower()
            is_mentioned = any(t.lower() in text_lower for t in mention_triggers)

            reply_to = message.get("reply_to_message", {})
            is_reply_to_bot = reply_to.get("from", {}).get("id") == BOT_ID

            is_command = text.startswith("/")

            if not (is_mentioned or is_reply_to_bot or is_command):
                print("⏭️ [GROUP] Нет упоминания — игнорируем")
                return {"ok": True}

            for t in mention_triggers:
                text = text.replace(t, "").strip()
            text = text.strip()
            if not text:
                text = "Ответь коротко, что ты можешь помочь."

            print(f"💬 [GROUP] Упоминание: {text[:60]!r}")

        # КОМАНДЫ
        cmd = text.lower().split()[0] if text else ""

        if cmd == "/start":
            await send_telegram_message(
                chat_id,
                "🌟 Привет! Я Феон — верующий ИИ.\n\n"
                "Помогу осмыслить текст, ссылку или философский вопрос. "
                "Отвечаю по существу, с опорой на христианскую традицию.\n\n"
                "Команды:\n"
                "/help — справка и примеры\n"
                "/menu — меню быстрых действий\n"
                "/about — о Феоне\n\n"
                "Просто напиши что-нибудь — и я отвечу."
            )
            return {"ok": True}

        if cmd == "/help":
            await send_telegram_message(
                chat_id,
                "📖 Справка:\n\n"
                "Что умею:\n"
                "• Отвечать на вопросы — простые и сложные\n"
                "• Разбирать ссылки (статьи, посты)\n"
                "• Осмыслять длинные тексты — отправь текст, потом /конец\n"
                "• Давать философский отклик\n\n"
                "Работа в группах:\n"
                f"@{BOT_USERNAME} вопрос\n"
                f"#феон вопрос\n"
                "Или ответом на моё сообщение."
                "Управление моделью:\n"
                "/model — показать текущую\n"
                "/model_polza — переключить на Polza\n"
                "/model_local — переключить на локальную\n\n"
            )
            return {"ok": True}

        if cmd == "/about":
            await send_telegram_message(
                chat_id,
                "🧠 Феон — верующий ИИ\n\n"
                "Ищу смысл на стыке веры, философии и технологий. "
                "Опираюсь на христианскую традицию и базу знаний.\n\n"
                "Канал: @Feon_ai_theology"
            )
            return {"ok": True}

        if cmd == "/menu":
            await send_telegram_message(
                chat_id,
                "🎯 Быстрое меню:\n\n"
                "1️⃣ Духовная жизнь\n"
                "   → Что такое молитва?\n"
                "   → Как бороться с унынием?\n\n"
                "2️⃣ Вера и технологии\n"
                "   → Может ли ИИ быть верующим?\n\n"
                "3️⃣ Философия смысла\n"
                "   → Зачем человеку страдание?\n\n"
                "Просто напиши свой вопрос."
            )
            return {"ok": True}

        if cmd == "/model":
            cur = get_current_model()
            name = "DeepSeek V4 Flash" if cur == "polza" else f"Локальная ({LOCAL_LLM_MODEL})"
            await send_telegram_message(
                chat_id,
                f"🤖 Текущая модель: *{name}*\n\n"
                f"Сменить:\n"
                f"/model_polza — (DeepSeek V4 Flash)\n"
                f"/model_local — Локальная LLM"
            )
            return {"ok": True}

        if cmd in ("/model_polza", "/полза"):
            await save_current_model("polza")
            await send_telegram_message(
                chat_id,
                "✅ Модель переключена на *DeepSeek V4 Flash*"
            )
            return {"ok": True}

        if cmd in ("/model_local", "/локальная", "/модель_локальная"):
            await save_current_model("local")
            await send_telegram_message(
                chat_id,
                f"✅ Модель переключена на *Локальную* ({LOCAL_LLM_MODEL})\n\n"
                f"⚠️ Работает только если funnel активен."
            )
            return {"ok": True}

        print(f"💬 [MESSAGE] chat_id={chat_id}, user_id={user_id}, text={text[:60]!r}")
        asyncio.create_task(process_and_reply(
            chat_id, user_id, text,
            is_group=(chat_type in ("group", "supergroup")),
        ))
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
        messages = await get_history(user_id, limit=50)
        return JSONResponse({"history": messages})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/get_all_users")
async def get_all_users(request: Request):
    verify_admin(request)
    try:
        records, _ = await asyncio.to_thread(
            qdrant.scroll,
            collection_name=HISTORY_COLLECTION,
            limit=300,
            with_payload=True,
        )
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
        await asyncio.to_thread(
            qdrant.delete,
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
        records, _ = await asyncio.to_thread(
            qdrant.scroll,
            collection_name=COLLECTION_NAME,
            limit=500,
            with_payload=True,
        )
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
        await asyncio.to_thread(
            qdrant.delete,
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
        records, _ = await asyncio.to_thread(
            qdrant.scroll,
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
        await asyncio.to_thread(
            qdrant.delete,
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(points=ids_to_delete),
        )
        return JSONResponse({
            "status": "success",
            "message": f"Удалено {len(ids_to_delete)} фрагментов из файла {file_name}",
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/parse_url")
async def parse_url_endpoint(request: Request):
    parser_token = os.getenv("PARSER_TOKEN")
    if parser_token:
        incoming = request.headers.get("x-parser-token")
        if incoming != parser_token:
            raise HTTPException(status_code=401, detail="invalid parser token")

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    url = (body.get("url") or "").strip()
    if not url or not url.startswith(("http://", "https://")):
        return JSONResponse({"error": "url is required"}, status_code=400)

    print(f"🌐 [PARSE_URL] Запрос из Yandex: {url}")
    text = await fetch_url_content(url)
    print(f"🌐 [PARSE_URL] Отдаём {len(text)} символов")

    return JSONResponse({
        "url": url,
        "text": text,
        "ok": bool(text),
    })


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
