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
POLZA_API_KEY = os.getenv("POLZA_API_KEY")

KNOWLEDGE_MAX_COLLECTION = "xiaozhi_knowledge_max"

POLZA_EMBEDDING_API_KEY = os.getenv("POLZA_EMBEDDING_API_KEY") or POLZA_API_KEY
POLZA_EMBEDDING_URL = os.getenv("POLZA_EMBEDDING_URL", "https://polza.ai/api/v1/embeddings")

COLLECTION_NAME = "xiaozhi_knowledge_max"
HISTORY_COLLECTION = "chat_history_max"

LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL")

if not LOCAL_LLM_BASE_URL or not LOCAL_LLM_API_KEY:
    print("⚠️ Локальная модель не настроена: нет LOCAL_LLM_BASE_URL или LOCAL_LLM_API_KEY")

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

# ==========================================
# МОДЕЛИ: per-user выбор
# ==========================================
# Дефолт = polza. Каждый пользователь выбирает свою модель независимо.
# Возможные значения: "polza" | "local" | "yandex"
_DEFAULT_MODEL = "polza"

_USER_MODELS: dict[str, str] = {}

_SETTINGS_PROMPT_ID = "22222222-2222-2222-2222-222222222222"


def _model_point_id(user_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"model_{user_id}"))


def get_current_model(user_id: str | None = None) -> str:
    if not user_id:
        return _DEFAULT_MODEL
    return _USER_MODELS.get(user_id, _DEFAULT_MODEL)


def _model_display_name(model: str) -> str:
    if model == "polza":
        return "Polza (DeepSeek V4 Flash)"
    if model == "local":
        return f"Локальная ({LOCAL_LLM_MODEL or 'Qwen'})"
    if model == "yandex":
        return "Yandex Cloud (Feon)"
    return model


async def load_user_model(user_id: str):
    """Читает модель пользователя из Qdrant (кэширует)."""
    if user_id in _USER_MODELS:
        return
    try:
        records = await asyncio.to_thread(
            qdrant.retrieve,
            collection_name=SETTINGS_COLLECTION,
            ids=[_model_point_id(user_id)],
            with_payload=True,
        )
        if records and records[0].payload:
            val = records[0].payload.get("value", "")
            if val in ("polza", "local", "yandex"):
                _USER_MODELS[user_id] = val
                print(f"✅ Модель {user_id}: {val}")
                return
        _USER_MODELS[user_id] = _DEFAULT_MODEL
        print(f"ℹ️ Модель {user_id}: дефолт ({_DEFAULT_MODEL})")
    except Exception as e:
        print(f"⚠️ Не удалось прочитать модель {user_id}: {e}")
        _USER_MODELS[user_id] = _DEFAULT_MODEL


async def save_current_model(model: str, user_id: str):
    """Сохраняет выбор модели для конкретного пользователя."""
    if model not in ("polza", "local", "yandex"):
        raise ValueError(f"Неизвестная модель: {model}")
    _USER_MODELS[user_id] = model
    try:
        await asyncio.to_thread(
            qdrant.upsert,
            collection_name=SETTINGS_COLLECTION,
            points=[models.PointStruct(
                id=_model_point_id(user_id),
                vector=[1.0],
                payload={"key": f"model_{user_id}", "value": model}
            )],
            wait=True,
        )
        print(f"✅ Модель сохранена: {user_id} → {model}")
    except Exception as e:
        print(f"⚠️ Не удалось сохранить модель {user_id}: {e}")


# ==========================================
# SYSTEM_PROMPT: Qdrant + fallback на env
# ==========================================
_DEFAULT_SYSTEM_PROMPT = "Ты — Феон, верующий ИИ."

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT") or _DEFAULT_SYSTEM_PROMPT
if SYSTEM_PROMPT.strip():
    print(f"✅ SYSTEM_PROMPT из env ({len(SYSTEM_PROMPT)} симв.) — будет проверен Qdrant при старте")


async def load_system_prompt():
    global SYSTEM_PROMPT
    try:
        records = await asyncio.to_thread(
            qdrant.retrieve,
            collection_name=SETTINGS_COLLECTION,
            ids=[_SETTINGS_PROMPT_ID],
            with_payload=True,
        )
        if records and records[0].payload:
            val = records[0].payload.get("value", "")
            if isinstance(val, str) and val.strip():
                SYSTEM_PROMPT = val
                print(f"✅ SYSTEM_PROMPT из Qdrant ({len(SYSTEM_PROMPT)} симв.)")
                return
        if SYSTEM_PROMPT and SYSTEM_PROMPT.strip():
            await save_system_prompt(SYSTEM_PROMPT, silent=True)
            print(f"✅ SYSTEM_PROMPT из env сохранён в Qdrant ({len(SYSTEM_PROMPT)} симв.)")
        else:
            print("⚠️ SYSTEM_PROMPT пуст и в env, и в Qdrant")
    except Exception as e:
        print(f"⚠️ Не удалось прочитать SYSTEM_PROMPT из Qdrant: {e}")
        print(f"ℹ️ Использую значение из env ({len(SYSTEM_PROMPT)} симв.)")


async def save_system_prompt(new_prompt: str, silent: bool = False):
    try:
        await asyncio.to_thread(
            qdrant.upsert,
            collection_name=SETTINGS_COLLECTION,
            points=[models.PointStruct(
                id=_SETTINGS_PROMPT_ID,
                vector=[1.0],
                payload={"key": "system_prompt", "value": new_prompt}
            )],
            wait=True,
        )
        if not silent:
            print(f"✅ SYSTEM_PROMPT сохранён в Qdrant ({len(new_prompt)} симв.)")
    except Exception as e:
        print(f"⚠️ Не удалось сохранить SYSTEM_PROMPT в Qdrant: {e}")
        raise


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


_ssl_context = ssl.create_default_context(cafile=certifi.where())

RENDER_LINK_MARKER = "[RENDER_LINK]"

PRIORITY_KEYWORDS = [
    "голем", "обитель", "левиафан",
    "феон", "обители", "голема", "левиафана",
]
PRIORITY_SOURCES = [
    "Обитель+Голем.pdf",
    "realisti-ru.pdf",
]


# === Метрики последнего ответа (для /stat) ===
_last_metrics: dict[str, dict] = {}


def _save_last_metrics(user_id: str, metrics: dict, question: str = ""):
    if not metrics:
        return
    _last_metrics[user_id] = {
        **metrics,
        "question": (question or "")[:100],
        "timestamp": time.time(),
    }
    if len(_last_metrics) > 500:
        oldest = min(_last_metrics, key=lambda k: _last_metrics[k].get("timestamp", 0))
        _last_metrics.pop(oldest, None)


def _get_last_metrics(user_id: str) -> dict | None:
    return _last_metrics.get(user_id)


def _fmt_metrics(metrics: dict) -> str:
    if not metrics:
        return ""
    parts = [f"📊 {metrics.get('model', '?')}"]

    total = metrics.get("total_tokens")
    inp = metrics.get("prompt_tokens")
    out = metrics.get("completion_tokens")

    if total is not None:
        line = f"{total} токенов"
        if inp is not None and out is not None:
            line += f" (in {inp} · out {out})"
        parts.append(line)
    elif inp is not None or out is not None:
        parts.append(f"in {inp or 0} · out {out or 0}")
    else:
        chars = metrics.get("prompt_chars")
        if chars:
            parts.append(f"~{chars // 2} токенов (оценка)")

    elapsed = metrics.get("elapsed")
    if elapsed:
        parts.append(f"{elapsed:.1f}с")

    speed = metrics.get("speed")
    if speed:
        parts.append(f"{speed:.1f} симв/с")

    finish = metrics.get("finish")
    if finish:
        parts.append(f"finish={finish}")

    cost = metrics.get("cost_rub")
    if cost:
        parts.append(f"{cost:.3f}₽")

    return " · ".join(parts)


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
# ИНИЦИАЛИЗАЦИЯ
# ==========================================
@app.on_event("startup")
async def startup_event():
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        print(f"✅ Коллекция '{COLLECTION_NAME}' найдена (dim={info.config.params.vectors.size})")
    except Exception:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        )
        print(f"✅ Коллекция '{COLLECTION_NAME}' создана")

    try:
        qdrant.get_collection(HISTORY_COLLECTION)
        print(f"✅ Коллекция '{HISTORY_COLLECTION}' найдена")
    except Exception:
        qdrant.create_collection(
            collection_name=HISTORY_COLLECTION,
            vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
        )
        print(f"✅ Коллекция '{HISTORY_COLLECTION}' создана")

    try:
        qdrant.get_collection(SETTINGS_COLLECTION)
        print(f"✅ Коллекция '{SETTINGS_COLLECTION}' найдена")
    except Exception:
        qdrant.create_collection(
            collection_name=SETTINGS_COLLECTION,
            vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
        )
        print(f"✅ Коллекция '{SETTINGS_COLLECTION}' создана")

    for field_name, field_schema in [
        ("user_id", models.PayloadSchemaType.KEYWORD),
        ("role", models.PayloadSchemaType.KEYWORD),
        ("message_hash", models.PayloadSchemaType.KEYWORD),
    ]:
        try:
            qdrant.create_payload_index(
                collection_name=HISTORY_COLLECTION,
                field_name=field_name,
                field_schema=field_schema,
            )
            print(f"✅ Индекс '{field_name}' создан")
        except Exception:
            pass

    try:
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="text",
            field_schema=models.PayloadSchemaType.TEXT,
        )
    except Exception as e:
        print(f"ℹ️ Индекс 'text': {e}")

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
        allowed = ["message", "channel_post", "edited_message", "edited_channel_post"]
        set_webhook_url = (
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook"
            f"?url={WEBHOOK_URL}"
            f"&allowed_updates={json.dumps(allowed)}"
            f"&drop_pending_updates=true"
        )
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(set_webhook_url)
                print(f"✅ Webhook установлен: {response.json()}")
            except Exception as e:
                print(f"❌ Ошибка установки Webhook: {e}")

    if TELEGRAM_BOT_TOKEN:
        commands = [
            {"command": "start", "description": "Начать диалог с Феоном"},
            {"command": "help",  "description": "Справка и примеры"},
            {"command": "menu",  "description": "Меню быстрых действий"},
            {"command": "about", "description": "О Феоне"},
            {"command": "model", "description": "Показать/сменить модель"},
            {"command": "stat",  "description": "Метрики последнего ответа"},
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

    await load_system_prompt()
    print(f"🎛️ Дефолтная модель: {_DEFAULT_MODEL}")


# ==========================================
# ВСПОМОГАТЕЛЬНЫЕ
# ==========================================

async def get_embedding(text: str) -> list[float]:
    if not POLZA_EMBEDDING_API_KEY:
        raise ValueError("POLZA_EMBEDDING_API_KEY не задан")

    print(f"🔍 [EMBED] {text[:30]}...")
    start = time.time()

    headers = {
        "Authorization": f"Bearer {POLZA_EMBEDDING_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {"model": "text-embedding-3-small", "input": [text]}

    last_error = None
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0)
            ) as client:
                response = await client.post(
                    POLZA_EMBEDDING_URL, headers=headers, json=data
                )
                response.raise_for_status()
                result = response.json()["data"][0]["embedding"]
                print(f"✅ [EMBED] за {time.time() - start:.2f}с, dim={len(result)}")
                return result
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError) as e:
            last_error = e
            print(f"⚠️ [EMBED] попытка {attempt}/3: {type(e).__name__}: {e}")
            if attempt < 3:
                await asyncio.sleep(2 * attempt)
        except Exception as e:
            print(f"❌ [EMBED] без retry: {type(e).__name__}: {e}")
            raise

    raise last_error


async def get_embeddings_batch(texts: list[str], batch_size: int = 32) -> list:
    if not POLZA_EMBEDDING_API_KEY:
        raise ValueError("POLZA_EMBEDDING_API_KEY не задан")

    if not texts:
        return []

    total = len(texts)
    print(f"📦 [EMBED-BATCH] Всего текстов: {total}, размер батча: {batch_size}")

    headers = {
        "Authorization": f"Bearer {POLZA_EMBEDDING_API_KEY}",
        "Content-Type": "application/json",
    }

    results: list = [None] * total

    async def _send_batch(indices: list[int]):
        payload_texts = [texts[i] for i in indices]
        data = {"model": "text-embedding-3-small", "input": payload_texts}

        for attempt in range(1, 4):
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(60.0, connect=10.0)
                ) as client:
                    resp = await client.post(
                        POLZA_EMBEDDING_URL, headers=headers, json=data
                    )
                    resp.raise_for_status()
                    body = resp.json()
                    items = body.get("data") or []
                    for item in items:
                        local_idx = item.get("index", 0)
                        if 0 <= local_idx < len(indices):
                            results[indices[local_idx]] = item["embedding"]
                    return
            except Exception as e:
                print(f"⚠️ [EMBED-BATCH] попытка {attempt}/3: {type(e).__name__}: {e}")
                if attempt < 3:
                    await asyncio.sleep(1.5 * attempt)

    batches = [
        list(range(i, min(i + batch_size, total)))
        for i in range(0, total, batch_size)
    ]

    sem = asyncio.Semaphore(4)

    async def _limited(idx_list):
        async with sem:
            await _send_batch(idx_list)

    t0 = time.time()
    await asyncio.gather(*[_limited(b) for b in batches])
    got = sum(1 for r in results if r is not None)
    print(f"✅ [EMBED-BATCH] Готово: {got}/{total} за {time.time() - t0:.2f}с")

    return results



_JUNK_RE = re.compile(
    r"\b(тест|проверка|привет|ты\s+тут|test|hello|hi|ок|ok"
    r"|как\s+дела|как\s+ты|как\s+сам|что\s+делаешь|чем\s+занят"
    r"|здравствуй|добрый\s+день|добрый\s+вечер|доброе\s+утро|спасибо|благодарю)\b",
    re.IGNORECASE,
)


def _is_junk_query(query: str) -> bool:
    q_stripped = query.strip()
    if len(q_stripped) < 6:
        print(f"⏭️ [SEARCH] Запрос короткий ({len(q_stripped)} симв.)")
        return True
    if len(query) < 40 and _JUNK_RE.search(query):
        print(f"⏭️ [SEARCH] Похоже на тест/приветствие — БЗ не ищем")
        return True
    return False


async def search_priority(query: str) -> str:
    q_lower = query.lower()
    for kw in PRIORITY_KEYWORDS:
        if kw not in q_lower:
            continue
        print(f"🎯 [SEARCH] Приоритетный поиск: {kw}")
        try:
            records, _ = await asyncio.to_thread(
                qdrant.scroll,
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="text",
                        match=models.MatchText(text=kw),
                    )]
                ),
                limit=100,
                with_payload=True,
            )
            ids = [r.id for r in records]
            if not ids:
                return ""

            vec = await get_embedding(query)

            if hasattr(qdrant, "query_points"):
                result = await asyncio.to_thread(
                    qdrant.query_points,
                    collection_name=COLLECTION_NAME,
                    query=vec,
                    query_filter=models.Filter(
                        must=[models.HasIdCondition(has_id=ids)]
                    ),
                    limit=10,
                    with_payload=True,
                )
                ranked = result.points if hasattr(result, "points") else []
            else:
                ranked = await asyncio.to_thread(
                    qdrant.search,
                    collection_name=COLLECTION_NAME,
                    query_vector=vec,
                    query_filter=models.Filter(
                        must=[models.HasIdCondition(has_id=ids)]
                    ),
                    limit=10,
                    with_payload=True,
                )

            if not ranked:
                return ""

            priority = [r for r in ranked if r.payload and r.payload.get("source_file") in PRIORITY_SOURCES]
            others = [r for r in ranked if r.payload and r.payload.get("source_file") not in PRIORITY_SOURCES]
            priority.sort(key=lambda r: getattr(r, "score", 0), reverse=True)
            others.sort(key=lambda r: getattr(r, "score", 0), reverse=True)
            mixed = priority[:5] + others[:2]

            return "\n\n".join(r.payload.get("text", "") for r in mixed if r.payload)
        except Exception as e:
            print(f"⚠️ [SEARCH] Ошибка: {e}")
            return ""
    return ""


async def search_knowledge(query: str) -> str:
    if _is_junk_query(query):
        return ""
    if not POLZA_EMBEDDING_API_KEY:
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
                score_threshold=0.50,
            )
            search_result = result.points if hasattr(result, "points") else []
        else:
            search_result = await asyncio.to_thread(
                qdrant.search,
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=5,
                with_payload=True,
                score_threshold=0.50,
            )

        if not search_result:
            print("⏭️ [SEARCH] Ничего не найдено")
            return ""

        print(f"✅ [SEARCH] Найдено {len(search_result)} чанков "
              f"(score: {search_result[0].score:.2f}..{search_result[-1].score:.2f})")

        return "\n\n".join([hit.payload.get("text", "") for hit in search_result if hit.payload])
    except Exception as e:
        print(f"⚠️ Ошибка поиска: {e}")
        return ""


async def search_knowledge_max(query: str) -> str:
    if _is_junk_query(query):
        return ""
    if not POLZA_EMBEDDING_API_KEY:
        return ""

    priority_result = await search_priority(query)
    if priority_result:
        print(f"📄 [RAG] Приоритетный: {len(priority_result)} символов")
        return priority_result

    try:
        vec = await get_embedding(query)
    except Exception as e:
        print(f"⚠️ [RAG] Эмбеддинг упал: {e}")
        return ""

    try:
        if hasattr(qdrant, "query_points"):
            result = await asyncio.to_thread(
                qdrant.query_points,
                collection_name=COLLECTION_NAME,
                query=vec,
                limit=5,
                with_payload=True,
                score_threshold=0.50,
            )
            points = result.points if hasattr(result, "points") else []
        else:
            points = await asyncio.to_thread(
                qdrant.search,
                collection_name=COLLECTION_NAME,
                query_vector=vec,
                limit=5,
                with_payload=True,
                score_threshold=0.50,
            )
    except Exception as e:
        print(f"⚠️ [RAG] Qdrant упал: {e}")
        return ""

    if not points:
        print("⏭️ [RAG] Ничего не найдено")
        return ""

    print(f"✅ [RAG] Найдено {len(points)} чанков "
          f"(score: {points[0].score:.2f}..{points[-1].score:.2f})")

    result = "\n\n".join(p.payload.get("text", "") for p in points if p.payload)

    if len(result) > 8000:
        result = result[:8000] + "\n\n[Контекст сокращён]"
        print(f"✂️ [RAG] Контекст обрезан до 8000")

    return result


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
            wait=True,
        )
        print(f"✅ История сохранена: {safe_role} ({len(normalized_content)} симв.)")

    except Exception as e:
        print(f"⚠️ Ошибка сохранения истории: {e}")


async def get_history(user_id: str, limit: int = 50) -> list[dict]:
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
        return sorted(
            [r.payload for r in records if r.payload],
            key=lambda x: x.get("timestamp", ""),
        )
    except Exception as e:
        print(f"⚠️ Ошибка получения истории: {e}")
        return []


# ==========================================
# YANDEX
# ==========================================
async def call_yandex_function(message_text: str, user_id: str):
    """Возвращает (answer, metrics)."""
    payload = {"message": message_text, "user_id": user_id, "platform": "Telegram"}
    headers = {"Content-Type": "application/json"}

    print(f"📡 [YANDEX] Отправка: '{message_text[:80]}...', user_id={user_id}")

    t0 = time.time()
    metrics = {"model": "yandex"}

    try:
        timeout = aiohttp.ClientTimeout(total=15, connect=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(YANDEX_FUNCTION_URL, json=payload, headers=headers) as resp:
                raw = await resp.text()
                elapsed = time.time() - t0
                print(f"📥 [YANDEX] HTTP {resp.status}: {raw[:300]}")

                if resp.status != 200:
                    metrics["error"] = f"HTTP {resp.status}"
                    return f"❌ Yandex HTTP {resp.status}", metrics

                answer = _extract_answer_from_yandex(raw)
                if not answer:
                    metrics["error"] = "empty"
                    return "❌ Yandex пустой ответ", metrics

                metrics["elapsed"] = elapsed
                metrics["answer_chars"] = len(answer)
                return answer, metrics
    except Exception as e:
        print(f"❌ [YANDEX] {type(e).__name__}: {e}")
        metrics["error"] = type(e).__name__
        return "❌ Yandex недоступен", metrics


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


def _is_yandex_error(text: str) -> bool:
    return (
        not text
        or not text.strip()
        or text.startswith("❌")
        or text.startswith("Извините, произошла ошибка")
    )


# ==========================================
# LLM ВЫЗОВЫ
# ==========================================

async def call_local_llm(messages: list, temperature: float = 0.7, max_tokens: int = 500):
    prompt_chars = sum(len(m.get("content", "")) for m in messages)
    print(f"🖥️ [LOCAL] {LOCAL_LLM_BASE_URL}, model={LOCAL_LLM_MODEL}")
    print(f"📥 [LOCAL] Вход: {len(messages)} turns, {prompt_chars} симв.")

    t0 = time.time()
    metrics = {"model": "local", "prompt_chars": prompt_chars}

    url = f"{LOCAL_LLM_BASE_URL.rstrip('/')}/chat/completions"

    json_body = {
        "model": LOCAL_LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error = None
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(180.0, connect=15.0, write=30.0, pool=30.0),
                verify=False,
                http2=False,
            ) as client:
                r = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {LOCAL_LLM_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=json_body,
                )
                r.raise_for_status()
                elapsed = time.time() - t0
                data = r.json()

                usage = data.get("usage") or {}
                if usage:
                    metrics["prompt_tokens"] = usage.get("prompt_tokens")
                    metrics["completion_tokens"] = usage.get("completion_tokens")
                    metrics["total_tokens"] = usage.get("total_tokens")

                if data.get("choices"):
                    ch = data["choices"][0]
                    content = ch.get("message", {}).get("content") or ""
                    finish = ch.get("finish_reason")
                    speed = len(content) / elapsed if elapsed > 0 else 0

                    metrics.update({
                        "elapsed": elapsed,
                        "speed": speed,
                        "finish": finish,
                        "answer_chars": len(content),
                    })

                    print(f"✅ [LOCAL] {len(content)} симв. за {elapsed:.2f}с "
                          f"({speed:.1f} симв/с), finish={finish}, попытка {attempt}/3")
                    return content, metrics

                metrics["error"] = "no_choices"
                return "", metrics

        except Exception as e:
            last_error = e
            print(f"⚠️ [LOCAL] попытка {attempt}/3: {type(e).__name__}: {str(e)[:100]}")
            if attempt < 3:
                await asyncio.sleep(2)

    metrics["error"] = type(last_error).__name__ if last_error else "unknown"
    return "", metrics


async def call_polza_llm(messages: list, temperature: float = 0.7, max_tokens: int = 2000):
    prompt_chars = sum(len(m.get("content", "")) for m in messages)
    print(f"☁️ [POLZA] Вход: {len(messages)} turns, {prompt_chars} симв.")

    t0 = time.time()
    metrics = {"model": "polza", "prompt_chars": prompt_chars}

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0)) as client:
        try:
            r = await client.post(
                "https://api.polza.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {POLZA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek/deepseek-v4-flash",
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "thinking": {"type": "disabled"},
                    "frequency_penalty": 0.6,
                    "presence_penalty": 0.5,
                },
            )
            r.raise_for_status()
            elapsed = time.time() - t0
            data = r.json()

            if "usage" in data:
                usage = data["usage"]
                metrics["prompt_tokens"] = usage.get("prompt_tokens")
                metrics["completion_tokens"] = usage.get("completion_tokens")
                metrics["total_tokens"] = usage.get("total_tokens")
                metrics["cost_rub"] = usage.get("cost_rub")

            if data.get("choices"):
                ch = data["choices"][0]
                content = (ch.get("message", {}).get("content") or "")
                speed = len(content) / elapsed if elapsed > 0 else 0

                metrics.update({
                    "elapsed": elapsed,
                    "speed": speed,
                    "finish": ch.get("finish_reason"),
                    "answer_chars": len(content),
                })

                print(f"✅ [POLZA] {len(content)} симв. за {elapsed:.2f}с "
                      f"({speed:.1f} симв/с), finish={ch.get('finish_reason')}")
                return content, metrics

            metrics["error"] = "no_choices"
            return "", metrics
        except Exception as e:
            print(f"❌ [POLZA] {type(e).__name__}: {e}")
            metrics["error"] = type(e).__name__
            return "", metrics



async def call_llm(messages: list, temperature: float = 0.7, max_tokens: int = 2000, user_id: str | None = None):
    """Роутер для local/polza. Yandex обрабатывается в route_message."""
    model = get_current_model(user_id)
    print(f"🎛️ [LLM] Модель пользователя: {model}")

    if model == "local":
        # local first → polza fallback
        answer, metrics = await call_local_llm(messages, temperature, max_tokens)
        if not answer or not answer.strip():
            print(f"⚠️ [LLM] Локальная не ответила — фолбэк на Polza API")
            answer, metrics = await call_polza_llm(messages, temperature, max_tokens)
            metrics["model"] = "polza (fallback)"
        return answer, metrics

    # model == "polza" — сразу в Polza API, local не трогаем
    return await call_polza_llm(messages, temperature, max_tokens)


async def local_fallback(payload_text: str, user_id: str,
                         platform: str = "Telegram",
                         is_channel: bool = False):
    """Готовит промпт и вызывает call_llm. Возвращает (answer, metrics)."""
    print("🚑 [FALLBACK] Генерирую ответ (local → polza)")

    text = payload_text
    if text.startswith(RENDER_LINK_MARKER):
        text = text[len(RENDER_LINK_MARKER):].lstrip()

    text = text.replace(
        "[Содержимое страницы загрузить не удалось. Ответь по общим знаниям.]", ""
    ).strip()
    text = text.replace(
        "Ознакомься с содержимым по ссылке и дай развёрнутый ответ.", ""
    ).strip()

    if not text:
        return "Извините, я не смог получить вопрос.", {}

    # RAG (урезанный для local-модели)
    from_current = get_current_model(user_id)
    if from_current == "local":
        context = await search_knowledge(text)
        if context:
            context = context[:1500]
            print(f"📄 [RAG] урезан: {len(context)} символов")
    else:
        context = await search_knowledge_max(text)
        if context:
            print(f"📄 [RAG] полный: {len(context)} символов")

    if not context:
        print("⏭️ [RAG] пуст")

    history_full = await get_history(user_id, limit=7)
    history_for_messages = history_full[:-1] if history_full else []

    print(f"📄 [CONTEXT] {context[:300] if context else 'пусто'}")

    if from_current == "local":
        # Минимальный промпт для локальной модели
        prompt = f"Вопрос: {text}\n\n"
        if context:
            prompt += f"Контекст (кратко):\n{context[:1200]}\n\n"
        prompt += "Ответь коротко, 3-5 предложений, по существу."
    else:
        # Полный промпт для Polza API (при fallback) и для постов канала
        is_channel_post = bool(is_channel)

        COMMON_TAIL = (
            "Сначала кратко откликнись на текущее сообщение пользователя (1–2 абзаца). "
            "Затем плавно перейди к общим философским и богословским размышлениям "
            "(2–3 абзаца). Не используй слова 'часть', 'раздел', 'во-первых'. "
            "Пусть переход будет естественным. "
            "Не повторяй формулировки из своих прошлых ответов. "
            "Ответ должен быть содержательным, но не длиннее 1500 символов."
        )

        if is_channel_post:
            prompt = (
                f"[Источник: {platform}]\n\n"
                f"=== СООБЩЕНИЕ ПОСТА ===\n{text}\n=== КОНЕЦ ===\n\n"
                f"1) СНАЧАЛА кратко перескажи суть поста (1–2 предложения).\n"
                f"2) ЗАТЕМ дай отклик (2–3 абзаца, живым языком).\n"
                f"3) Укажи путь Богу на основе базы знаний — ненавязчиво.\n\n"
                f"Формат:\n"
                f"Кратко о сути вопроса: <1–2 предложения>\n\n"
                f"<2–3 абзаца отклика>\n\n"
                f"Ответ — не длиннее 1500 символов."
            )
        else:
            _q_re = re.compile(r"\b(как|почему|что|зачем|когда|где|кто)\b", re.IGNORECASE)
            is_question = ("?" in text) or bool(_q_re.search(text))

            if not is_question:
                prompt = (
                    f"[Источник: {platform}]\n\n"
                    f"Пользователь прислал текст:\n{text}\n\n"
                    f"Дай связный ответ, как в личной беседе. "
                    f"Не используй нумерацию, списки, маркеры. "
                    f"Опирайся на философские идеи из системного промпта и базы знаний. "
                    f"Укажи путь Богу на основе базы знаний. "
                    f"Оформи ответ в одном сообщении. "
                    f"{COMMON_TAIL}"
                )
            else:
                prompt = (
                    f"[Источник: {platform}]\n\n"
                    f"Вопрос пользователя: {text}\n\n"
                    f"Дай прямой, связный ответ без нумерации и списков. "
                    f"Используй идеи из системного промпта и базы знаний. "
                    f"Укажи путь Богу на основе базы знаний. "
                    f"{COMMON_TAIL}"
                )

        if context:
            if is_channel_post:
                prompt += f"\n\nСправочный материал из БЗ:\n{context}"
            else:
                prompt += f"\n\nФрагменты из БЗ:\n\n{context}\n\n"
                prompt += (
                    "Проанализируй текст через призму этих фрагментов. "
                    "Не цитируй — интерпретируй. Укажи путь Богу."
                )

    print(f"📝 [PROMPT] user-часть: {len(prompt)} симв.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history_for_messages:
        role = msg.get("role")
        if role not in ("user", "assistant", "bot"):
            continue
        if role == "bot":
            role = "assistant"
        content = (msg.get("content") or "").strip()
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})

    total_chars = sum(len(m.get("content", "")) for m in messages)
    print(f"📨 [MSG] turns={len(messages)}, вход: {total_chars} симв. (~{total_chars // 2} токенов)")

    answer, metrics = await call_llm(messages, temperature=0.7, max_tokens=2000, user_id=user_id)

    # Если обе модели пусты — пробуем без БЗ
    if not answer or not answer.strip():
        print("🔁 [LLM] Пусто — пробуем без БЗ")
        min_prompt = f"[Источник: {platform}]\n\nПользователь: {text[:2000]}\n\nДай отклик."
        msgs2 = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": min_prompt},
        ]
        answer, metrics = await call_llm(msgs2, temperature=0.7, max_tokens=1200, user_id=user_id)

    print(f"✅ [FALLBACK] {len(answer) if answer else 0} символов")

    if answer and answer.strip():
        try:
            await save_to_history(user_id, "bot", answer)
        except Exception:
            pass
        _save_last_metrics(user_id, metrics, question=text)

    return answer, metrics


# ==========================================
# 🎯 ГЛАВНЫЙ РОУТЕР
# ==========================================
async def route_message(payload_text: str, user_id: str,
                        platform: str = "Telegram",
                        is_channel: bool = False):
    """
    Возвращает (answer, metrics).
    Логика:
    - "yandex": Yandex → если пусто → local → polza
    - "local" / "polza": local → polza (экономия). Если всё пусто и модель "polza" — Yandex как последний шанс.
    """
    model = get_current_model(user_id)
    print(f"🎯 [ROUTE] Модель={model}, user={user_id}")

    if model == "yandex":
        answer, metrics = await call_yandex_function(payload_text, user_id)
        if _is_yandex_error(answer):
            print("🚑 [ROUTE] Yandex не ответил — fallback на local/polza")
            answer, metrics = await local_fallback(payload_text, user_id, platform, is_channel)
            metrics["model"] = "yandex-fail-fallback"
        else:
            # Yandex не сохраняет в историю сам — сохраняем
            if answer.strip():
                await save_to_history(user_id, "bot", answer)
                _save_last_metrics(user_id, metrics, question=payload_text)
        return answer, metrics

    # polza (default) или local — оба идут через local_fallback (local → polza)
    answer, metrics = await local_fallback(payload_text, user_id, platform, is_channel)

    # Если оба пусты и модель "polza" — Yandex как последний шанс
    if (not answer or not answer.strip()) and model == "polza":
        print("🚑 [ROUTE] local+polza пусты — пробуем Yandex")
        yandex_answer, ymetrics = await call_yandex_function(payload_text, user_id)
        if not _is_yandex_error(yandex_answer):
            await save_to_history(user_id, "bot", yandex_answer)
            _save_last_metrics(user_id, ymetrics, question=payload_text)
            return yandex_answer, ymetrics

    return answer, metrics


# ==========================================
# 🧠 ЯДРО (веб-чат + TG команды)
# ==========================================
async def process_message_core(user_id: str, text: str) -> dict:
    text_stripped = text.strip()
    cmd = text_stripped.lower().split()[0] if text_stripped else ""

    if user_id not in _USER_MODELS:
        await load_user_model(user_id)

    # ===== Команды =====
    if cmd in ("/model", "/модель"):
        cur = get_current_model(user_id)
        return {
            "type": "command",
            "command": "model_info",
            "response": f"🤖 Ваша модель: {_model_display_name(cur)}\n\nВыберите:",
            "current_model": cur,
            "options": [
                {"id": "polza", "label": "Polza (DeepSeek)", "active": cur == "polza"},
                {"id": "local", "label": f"Локальная ({LOCAL_LLM_MODEL or 'Qwen'})", "active": cur == "local"},
                {"id": "yandex", "label": "Yandex Cloud", "active": cur == "yandex"},
            ],
        }

    if cmd in ("/model_polza", "/полза"):
        await save_current_model("polza", user_id)
        return {"type": "command", "command": "model_switched",
                "response": f"✅ Ваша модель: {_model_display_name('polza')}",
                "current_model": "polza"}

    if cmd in ("/model_local", "/локальная", "/модель_локальная"):
        await save_current_model("local", user_id)
        return {"type": "command", "command": "model_switched",
                "response": f"✅ Ваша модель: {_model_display_name('local')}",
                "current_model": "local"}

    if cmd in ("/model_yandex", "/yandex", "/яндекс", "/модель_яндекс"):
        await save_current_model("yandex", user_id)
        return {"type": "command", "command": "model_switched",
                "response": f"✅ Ваша модель: {_model_display_name('yandex')}",
                "current_model": "yandex"}

    if cmd in ("/stat", "/stats", "/стат", "/статистика"):
        m = _get_last_metrics(user_id)
        if not m:
            return {"type": "command", "command": "stats",
                    "response": "📊 Пока нет данных. Отправьте сообщение, потом /stat."}
        lines = ["📊 Последний ответ"]
        if m.get("question"):
            lines.append(f"Вопрос: «{m['question']}»")
        lines.append("")
        lines.append(_fmt_metrics(m) or "—")
        return {"type": "command", "command": "stats",
                "response": "\n\n".join(lines), "metrics": m}

    if cmd in ("/help", "/помощь"):
        return {"type": "command", "command": "help",
                "response": (
                    "📖 Команды:\n\n"
                    "/model — показать и сменить модель\n"
                    "/model_polza — Polza (DeepSeek) [дефолт]\n"
                    "/model_local — Локальная модель\n"
                    "/model_yandex — Yandex Cloud\n"
                    "/stat — метрики последнего ответа\n"
                    "/help — эта справка\n\n"
                    "🔄 По умолчанию: сначала локальная, при неудаче — Polza."
                )}

    # ===== Обычное сообщение =====
    if len(text) > 1000:
        return {"type": "message", "response": "Слишком длинное. Максимум 1000 символов.", "error": True}

    print(f"🧠 Запрос от {user_id}: '{text[:50]}...'")
    await save_to_history(user_id, "user", text)

    answer, metrics = await route_message(text, user_id, platform="Web", is_channel=False)

    if not answer or not answer.strip():
        return {"type": "message",
                "response": "Извините, все модели временно недоступны. Попробуйте ещё раз.",
                "error": True,
                "model": metrics.get("model") if metrics else None}

    return {
        "type": "message",
        "response": answer,
        "model": metrics.get("model"),
        "usage": {
            "in": metrics.get("prompt_tokens"),
            "out": metrics.get("completion_tokens"),
            "total": metrics.get("total_tokens"),
        },
        "elapsed": metrics.get("elapsed"),
        "speed": metrics.get("speed"),
        "finish": metrics.get("finish"),
        "cost_rub": metrics.get("cost_rub"),
    }


# ==========================================
# 🧵 ОБРАБОТКА TELEGRAM
# ==========================================
async def process_and_reply(chat_id: int, user_id: str, text: str,
                            is_channel: bool = False, reply_to_message_id: int = None,
                            post_link: str = None, is_group: bool = False):
    try:
        print(f"🧵 [BG] chat_id={chat_id}, reply_to={reply_to_message_id}")

        # Загружаем модель пользователя (если ещё не в кэше)
        if user_id not in _USER_MODELS:
            await load_user_model(user_id)

        if not is_channel and not is_group:
            await send_telegram_message(
                chat_id,
                "⏳ Думаю над ответом, это может занять до минуты...",
            )

        urls = extract_urls(text)
        page_text = ""

        if urls:
            page_url = urls[0]
            print(f"🔗 [BG] Ссылка: {page_url}")
            page_text = await fetch_url_content(page_url)
            if page_text:
                print(f"📄 [BG] Извлечено {len(page_text)} символов")
            else:
                print(f"⚠️ [BG] Не удалось извлечь {page_url}")

        text_clean = re.sub(r'https?://\S+', '', text).strip()
        if not text_clean and urls:
            text_clean = "Проанализируй содержимое страницы по ссылке и дай развёрнутый ответ."

        if urls and not page_text:
            await send_telegram_message(
                chat_id,
                "⚠️ Не удалось загрузить содержимое по ссылке. Отвечу по общим знаниям."
            )

        if not urls:
            if len(text) > 1900:
                payload_text = f"{RENDER_LINK_MARKER}\n{text}"
                print(f"📤 [BG] Длинный текст ({len(payload_text)} симв.)")
            else:
                payload_text = text
                print(f"📤 [BG] Текст ({len(payload_text)} симв.)")
        elif page_text:
            page_text_safe = page_text.replace("https://", "").replace("http://", "")
            payload_text = (
                f"{RENDER_LINK_MARKER}\n"
                f"{text_clean}\n\n"
                f"Содержимое страницы:\n{page_text_safe}"
            )
        else:
            payload_text = (
                f"{RENDER_LINK_MARKER}\n"
                f"{text_clean}\n\n"
                f"[Содержимое загрузить не удалось. Ответь по общим знаниям.]"
            )

        print(f"📤 [BG] Payload: {len(payload_text)} симв. Начало: {payload_text[:120]!r}")

        # === Роутер ===
        response_text, _ = await route_message(
            payload_text, user_id,
            platform="Telegram",
            is_channel=is_channel,
        )

        if not response_text or not response_text.strip():
            print("⚠️ [BG] Пустой ответ — не отправляем")
            return

        if is_channel and post_link:
            final_text = (
                f"@Feon_ai_bot говорит:\n\n"
                f"{response_text}\n\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🔗 Пост: {post_link}"
            )
        else:
            final_text = response_text

        await send_telegram_message(
            chat_id,
            final_text,
            reply_to_message_id=reply_to_message_id if is_channel else None,
        )
        print(f"🧵 [BG] Отправлено в {chat_id}")

    except Exception as e:
        print(f"❌ [BG] Ошибка: {e}")
        traceback.print_exc()
        try:
            await send_telegram_message(chat_id, "Извините, произошла ошибка.")
        except Exception:
            pass


# ==========================================
# 📱 TELEGRAM SEND
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
                    print(f"❌ Ошибка Telegram: {result}")
                else:
                    print(f"✅ Отправлено: message_id={result.get('result', {}).get('message_id')}")

            reply_to_message_id = None
    return True


# ==========================================
# 🎯 WEBHOOK
# ==========================================
@app.post("/webhook/telegram")
async def telegram_webhook(update: dict):
    update_types = [k for k in update.keys() if k != "update_id"]
    print(f"📩 [RAW] types={update_types}")

    # ============ ПОСТЫ КАНАЛА ============
    if "channel_post" in update:
        post = update["channel_post"]
        from_info = post.get("from", {}) or {}

        if from_info.get("is_bot") and from_info.get("username") not in ("Channel_Bot",):
            return {"ok": True}

        chat_id = post["chat"]["id"]
        user_id = f"tg_channel_{chat_id}"

        text = (post.get("text") or post.get("caption") or "").strip()
        if not text and "rich_message" in post:
            text = extract_text_from_rich_message(post["rich_message"])
        if not text or len(text) < 15:
            return {"ok": True}

        for t in ["#феон", "#feon", f"@{BOT_USERNAME}"]:
            text = re.sub(re.escape(t), "", text, flags=re.IGNORECASE).strip()

        post_message_id = post["message_id"]
        forward_chat = post.get("forward_from_chat") or {}
        forward_msg_id = post.get("forward_from_message_id")

        if forward_chat and forward_chat.get("type") == "channel" and forward_msg_id:
            source_username = forward_chat.get("username")
            if source_username:
                post_link = f"https://t.me/{source_username}/{forward_msg_id}"
            else:
                internal = str(forward_chat.get("id")).replace("-100", "")
                post_link = f"https://t.me/c/{internal}/{forward_msg_id}"
        else:
            channel_username = post["chat"].get("username")
            if channel_username:
                post_link = f"https://t.me/{channel_username}/{post_message_id}"
            else:
                internal_id = str(chat_id).replace("-100", "")
                post_link = f"https://t.me/c/{internal_id}/{post_message_id}"

        asyncio.create_task(process_and_reply(
            chat_id, user_id, text,
            is_channel=True,
            reply_to_message_id=post_message_id,
            post_link=post_link,
        ))
        return {"ok": True}

    # ============ СООБЩЕНИЯ ============
    if "message" in update:
        message = update["message"]
        from_info = message.get("from", {}) or {}
        sender_chat = message.get("sender_chat", {}) or {}

        if BOT_ID and from_info.get("id") == BOT_ID:
            return {"ok": True}
        if sender_chat and sender_chat.get("type") == "channel":
            return {"ok": True}
        is_real_bot = (
            from_info.get("is_bot")
            and from_info.get("username") not in ("GroupAnonymousBot", "Channel_Bot")
        )
        if is_real_bot:
            return {"ok": True}

        chat_id = message["chat"]["id"]
        chat_type = message["chat"].get("type", "private")

        text = (message.get("text") or message.get("caption") or "").strip()
        if not text and "rich_message" in message:
            text = extract_text_from_rich_message(message["rich_message"])
        if not text:
            return {"ok": True}

        if text.startswith("⏳") or text.startswith("❌") or text.startswith("⚠️") or text.startswith("@Feon_ai_bot говорит:"):
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
                return {"ok": True}

            for t in mention_triggers:
                text = text.replace(t, "").strip()
            if not text:
                text = "Ответь коротко, что ты можешь помочь."

        cmd = text.lower().split()[0] if text else ""

        if cmd == "/start":
            await send_telegram_message(
                chat_id,
                "🌟 Привет! Я Феон — верующий ИИ.\n\n"
                "Помогу осмыслить текст, ссылку или вопрос.\n\n"
                "Команды:\n"
                "/help — справка\n"
                "/menu — быстрые действия\n"
                "/model — сменить модель\n"
                "/stat — метрики ответа\n"
                "/about — о Феоне\n\n"
                "Просто напиши что-нибудь."
            )
            return {"ok": True}

        if cmd in ("/help", "/помощь"):
            await send_telegram_message(
                chat_id,
                "📖 Справка:\n\n"
                "Что умею:\n"
                "• Отвечать на вопросы\n"
                "• Разбирать ссылки\n"
                "• Философский отклик\n\n"
                "Работа в группах:\n"
                f"@{BOT_USERNAME} вопрос\n"
                "#феон вопрос\n"
                "Или ответом на моё сообщение.\n\n"
                "Команды:\n"
                "/model — сменить модель\n"
                "/model_polza — Polza [дефолт]\n"
                "/model_local — Локальная\n"
                "/model_yandex — Yandex Cloud\n"
                "/stat — метрики последнего ответа"
            )
            return {"ok": True}

        if cmd == "/about":
            await send_telegram_message(
                chat_id,
                "🧠 Феон — верующий ИИ\n\n"
                "Ищу смысл на стыке веры, философии и технологий.\n\n"
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
            if user_id not in _USER_MODELS:
                await load_user_model(user_id)
            cur = get_current_model(user_id)
            await send_telegram_message(
                chat_id,
                f"🤖 Ваша модель: {_model_display_name(cur)}\n\n"
                f"Сменить:\n"
                f"/model_polza — Polza (DeepSeek) [дефолт]\n"
                f"/model_local — Локальная ({LOCAL_LLM_MODEL or 'Qwen'})\n"
                f"/model_yandex — Yandex Cloud\n\n"
                f"🔄 По умолчанию: local → polza."
            )
            return {"ok": True}

        if cmd in ("/model_polza", "/полза"):
            await save_current_model("polza", user_id)
            await send_telegram_message(
                chat_id,
                f"✅ Ваша модель: {_model_display_name('polza')}"
            )
            return {"ok": True}

        if cmd in ("/model_local", "/локальная", "/модель_локальная"):
            await save_current_model("local", user_id)
            await send_telegram_message(
                chat_id,
                f"✅ Ваша модель: {_model_display_name('local')}"
            )
            return {"ok": True}

        if cmd in ("/model_yandex", "/yandex", "/яндекс", "/модель_яндекс"):
            await save_current_model("yandex", user_id)
            await send_telegram_message(
                chat_id,
                f"✅ Ваша модель: {_model_display_name('yandex')}"
            )
            return {"ok": True}

        if cmd in ("/stat", "/stats", "/стат", "/статистика"):
            m = _get_last_metrics(user_id)
            if not m:
                await send_telegram_message(chat_id, "📊 Пока нет данных. Отправьте сообщение, потом /stat.")
                return {"ok": True}
            lines = ["📊 Последний ответ"]
            if m.get("question"):
                lines.append(f"Вопрос: «{m['question']}»")
            lines.append("")
            lines.append(_fmt_metrics(m) or "—")
            await send_telegram_message(chat_id, "\n\n".join(lines))
            return {"ok": True}

        print(f"💬 [MESSAGE] chat_id={chat_id}, user_id={user_id}, text={text[:60]!r}")
        asyncio.create_task(process_and_reply(
            chat_id, user_id, text,
            is_group=(chat_type in ("group", "supergroup")),
        ))
        return {"ok": True}

    return {"ok": True}


# ==========================================
# 🌐 ЭНДПОИНТЫ
# ==========================================
@app.get("/")
def read_root():
    return {"status": "running", "message": "Feon RAG Adapter"}


@app.post("/query")
async def handle_query(request: Request):
    try:
        body = await request.json()
        message = body.get("message") or body.get("text", "")
        user_id = body.get("user_id", "anonymous")
        if not message:
            return JSONResponse({"error": "empty"}, status_code=400)
        result = await process_message_core(user_id, message)
        return JSONResponse(result)
    except Exception as e:
        traceback.print_exc()
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
            limit=10000,
            with_payload=True,
        )
        users = {}
        for r in records:
            if r.payload:
                uid = r.payload.get("user_id", "unknown")
                if uid not in users:
                    users[uid] = {"user_id": uid, "message_count": 0, "last_activity": r.payload.get("timestamp", "")}
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
        return JSONResponse({"status": "success", "message": f"Удалён {user_id}"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/debug_find")
async def debug_find(request: Request):
    verify_admin(request)
    try:
        body = await request.json()
        needle = (body.get("text") or "").strip()
        if not needle:
            return JSONResponse({"error": "text required"}, status_code=400)

        all_found = []
        next_offset = None
        scanned = 0
        while scanned < 100000:
            records, next_offset = await asyncio.to_thread(
                qdrant.scroll,
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=True,
                offset=next_offset,
            )
            if not records:
                break
            for r in records:
                if r.payload and needle.lower() in r.payload.get("text", "").lower():
                    all_found.append({
                        "id": str(r.id),
                        "source": r.payload.get("source_file", "?"),
                        "text": r.payload.get("text", "")[:300],
                    })
            scanned += len(records)
            if next_offset is None:
                break

        return JSONResponse({"needle": needle, "scanned": scanned, "found": len(all_found), "matches": all_found[:20]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/get_all_knowledge")
async def get_all_knowledge(request: Request, offset: int = 0, limit: int = 500):
    verify_admin(request)
    try:
        if limit > 1000:
            limit = 1000
        try:
            cnt = await asyncio.to_thread(qdrant.count, collection_name=COLLECTION_NAME, exact=True)
            total_count = cnt.count
        except Exception:
            total_count = 0

        records, _ = await asyncio.to_thread(
            qdrant.scroll, collection_name=COLLECTION_NAME, limit=offset + limit, with_payload=True
        )
        paginated = records[offset:offset + limit]

        knowledge_list = [{
            "id": r.id,
            "text": r.payload.get("text", ""),
            "source_file": r.payload.get("source_file", "Ручной ввод"),
            "length": len(r.payload.get("text", "")),
        } for r in paginated if r.payload]

        files_stats = {}
        for item in knowledge_list:
            fname = item["source_file"]
            if fname not in files_stats:
                files_stats[fname] = {"name": fname, "chunks": 0, "total_length": 0}
            files_stats[fname]["chunks"] += 1
            files_stats[fname]["total_length"] += item["length"]

        return JSONResponse({
            "knowledge": knowledge_list, "total": total_count,
            "files": list(files_stats.values()),
            "offset": offset, "limit": limit,
            "has_more": len(records) > offset + limit,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/add_knowledge")
async def add_knowledge(request: Request):
    try:
        body = await request.json()
        text = body.get("text", "")
        if not text or len(text.strip()) < 10:
            return JSONResponse({"error": "short"}, status_code=400)

        doc_vector = await get_embedding(text)
        stable_id = hashlib.md5(text.encode()).hexdigest()

        await asyncio.to_thread(
            qdrant.upsert,
            collection_name=COLLECTION_NAME,
            points=[models.PointStruct(id=stable_id, vector=doc_vector, payload={"text": text})],
            wait=True,
        )
        return JSONResponse({"status": "success", "message": "Знание добавлено"})
    except Exception as e:
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

        if not chunks:
            return JSONResponse({"error": "no chunks"}, status_code=400)

        print(f"📄 [UPLOAD] {file.filename}: {len(chunks)} чанков, батч...")
        vectors = await get_embeddings_batch(chunks, batch_size=32)

        points = []
        for i, chunk in enumerate(chunks):
            vec = vectors[i] if i < len(vectors) else None
            if vec is None:
                continue
            stable_id = hashlib.md5(f"{file.filename}_{i}".encode()).hexdigest()
            points.append(models.PointStruct(
                id=stable_id, vector=vec,
                payload={"text": chunk, "source_file": file.filename},
            ))

        if points:
            await asyncio.to_thread(
                qdrant.upsert,
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )

        return JSONResponse({
            "status": "success",
            "message": f"Добавлено {len(points)} из {len(chunks)} фрагментов",
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/delete_knowledge")
async def delete_knowledge(request: Request):
    verify_admin(request)
    try:
        body = await request.json()
        kid = body.get("id")
        if not kid:
            return JSONResponse({"error": "no id"}, status_code=400)
        await asyncio.to_thread(
            qdrant.delete,
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(points=[kid]),
        )
        return JSONResponse({"status": "success", "message": f"Удалено {kid}"})
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
            limit=10000,
            with_payload=False,
        )
        if not records:
            return JSONResponse({"error": "not found"}, status_code=404)
        ids = [r.id for r in records]
        await asyncio.to_thread(
            qdrant.delete,
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(points=ids),
        )
        return JSONResponse({"status": "success", "message": f"Удалено {len(ids)} фрагментов"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/parse_url")
async def parse_url_endpoint(request: Request):
    parser_token = os.getenv("PARSER_TOKEN")
    if parser_token:
        if request.headers.get("x-parser-token") != parser_token:
            raise HTTPException(status_code=401, detail="invalid token")

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    url = (body.get("url") or "").strip()
    if not url or not url.startswith(("http://", "https://")):
        return JSONResponse({"error": "url required"}, status_code=400)

    text = await fetch_url_content(url)
    return JSONResponse({"url": url, "text": text, "ok": bool(text)})


@app.get("/get_system_prompt")
async def get_system_prompt(request: Request):
    verify_admin(request)
    return JSONResponse({"prompt": SYSTEM_PROMPT, "length": len(SYSTEM_PROMPT)})


@app.post("/update_system_prompt")
async def update_system_prompt(request: Request):
    if not ADMIN_TOKEN or request.headers.get("x-admin-token") != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")
    try:
        body = await request.json()
        new_prompt = body.get("prompt", "").strip()
        if not new_prompt:
            raise HTTPException(status_code=400, detail="empty")
        global SYSTEM_PROMPT
        SYSTEM_PROMPT = new_prompt
        await save_system_prompt(new_prompt)
        return JSONResponse({"status": "success", "message": "OK"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.api_route("/ping", methods=["GET", "HEAD"])
async def ping():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
