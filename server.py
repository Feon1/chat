import os
import logging
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import datetime
import uvicorn
from qdrant_client import QdrantClient
from qdrant_client.http import models
# from sentence_transformers import SentenceTransformer
import numpy as np
from light_embed import TextEmbedding

# ---------- НАСТРОЙКА ЛОГГИРОВАНИЯ ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ----------
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "default_admin_token")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
VK_GROUP_TOKEN = os.getenv("VK_GROUP_TOKEN")
VK_GROUP_ID = os.getenv("VK_GROUP_ID")
MAX_BOT_TOKEN = os.getenv("MAX_BOT_TOKEN")
MAX_WEBHOOK_URL = os.getenv("MAX_WEBHOOK_URL")

# ---------- ПОДКЛЮЧЕНИЕ К QDRANT ----------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# ---------- МОДЕЛЬ ЭМБЕДДИНГОВ ----------

embedding_model = TextEmbedding('onnx-models/paraphrase-multilingual-MiniLM-L12-v2-onnx')

# ---------- FASTAPI APP ----------
app = FastAPI(title="XiaoZhi RAG Adapter")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ваша логика) ----------
def get_embedding(text: str) -> list:
    embedding = embedding_model.encode([text])[0]  # получаем numpy-массив
    return embedding.tolist()

async def search_knowledge(query: str, top_k: int = 5) -> list:
    query_vector = get_embedding(query)
    search_result = qdrant_client.search(
        collection_name="xiaozhi_knowledge",
        query_vector=query_vector,
        limit=top_k
    )
    return [hit.payload["text"] for hit in search_result]

async def save_chat_history(user_id: str, query: str, answer: str):
    timestamp = datetime.now().isoformat()
    point = {
        "user_id": user_id,
        "query": query,
        "answer": answer,
        "timestamp": timestamp
    }
    qdrant_client.upsert(
        collection_name="chat_history",
        points=[
            models.PointStruct(
                id=hash(f"{user_id}_{timestamp}"),
                vector=[0.0]*384,
                payload=point
            )
        ]
    )

async def process_message_core(user_id: str, text: str) -> str:
    """Основная логика обработки сообщения"""
    docs = await search_knowledge(text)
    context = "\n".join(docs) if docs else "Нет релевантных документов."
    answer = f"Пользователь {user_id} спросил: {text}\n\nКонтекст:\n{context}\n\nОтвет бота (заглушка)."
    await save_chat_history(user_id, text, answer)
    return answer

# ---------- ИНТЕГРАЦИЯ С TELEGRAM ----------
async def set_telegram_webhook():
    if not TELEGRAM_BOT_TOKEN or not WEBHOOK_URL:
        logger.warning("TELEGRAM_BOT_TOKEN или WEBHOOK_URL не заданы")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook?url={WEBHOOK_URL}/webhook/telegram"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        logger.info(f"Telegram Webhook установлен: {resp.json()}")

@app.post("/webhook/telegram")
async def telegram_webhook(request: Request):
    body = await request.json()
    logger.info(f"Telegram webhook: {body}")
    if "message" in body:
        chat_id = body["message"]["chat"]["id"]
        user_id = str(body["message"]["from"]["id"])
        text = body["message"].get("text", "")
        if text:
            answer = await process_message_core(user_id, text)
            send_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            async with httpx.AsyncClient() as client:
                await client.post(send_url, json={"chat_id": chat_id, "text": answer})
    return {"status": "ok"}

# ---------- ИНТЕГРАЦИЯ С VK ----------
@app.post("/webhook/vk")
async def vk_webhook(request: Request):
    body = await request.json()
    logger.info(f"VK webhook: {body}")
    return {"status": "ok"}

# ---------- ИНТЕГРАЦИЯ С MAX ----------
async def set_max_webhook():
    """Устанавливает вебхук для MAX бота"""
    if not MAX_BOT_TOKEN or not MAX_WEBHOOK_URL:
        logger.warning("MAX_BOT_TOKEN или MAX_WEBHOOK_URL не заданы, пропускаем")
        return

    url = "https://platform-api2.max.ru/webhook"
    headers = {
        "Authorization": MAX_BOT_TOKEN,
        "Content-Type": "application/json"
    }
    payload = {"url": MAX_WEBHOOK_URL}

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            logger.info(f"✅ MAX Webhook установлен: {response.json()}")
        except Exception as e:
            logger.error(f"❌ Ошибка установки MAX Webhook: {e}")

async def send_max_message(chat_id: str, text: str):
    """Отправляет сообщение в чат MAX"""
    if not MAX_BOT_TOKEN:
        logger.error("MAX_BOT_TOKEN не задан, сообщение не отправлено")
        return

    url = "https://platform-api2.max.ru/messages"
    headers = {
        "Authorization": MAX_BOT_TOKEN,
        "Content-Type": "application/json"
    }
    payload = {"chat_id": chat_id, "text": text}

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            logger.info(f"Сообщение отправлено в чат {chat_id}")
            return response.json()
        except Exception as e:
            logger.error(f"Ошибка отправки в MAX: {e}")

@app.post("/webhook/max")
async def max_webhook(request: Request):
    """Эндпоинт для приёма вебхуков от MAX"""
    try:
        body = await request.json()
        logger.info(f"MAX webhook: {body}")
    except Exception as e:
        logger.error(f"Ошибка парсинга JSON: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event_type = body.get("event_type")
    if event_type == "message":
        chat_id = body["chat"]["id"]
        user_id = str(body["from"]["id"])
        text = body.get("text", "")

        try:
            answer = await process_message_core(user_id, text)
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            answer = "Извините, произошла ошибка."

        await send_max_message(chat_id, answer)
        return {"status": "ok"}

    logger.info(f"Событие {event_type} не обрабатывается")
    return {"status": "ignored"}

# ---------- ЗАПУСК ПРИ СТАРТЕ ----------
@app.on_event("startup")
async def startup_event():
    """Выполняется при старте приложения"""
    # Проверка коллекций Qdrant
    collections = qdrant_client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if "xiaozhi_knowledge" not in collection_names:
        qdrant_client.create_collection(
            collection_name="xiaozhi_knowledge",
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )
        logger.info("Создана коллекция xiaozhi_knowledge")
    else:
        logger.info("Коллекция xiaozhi_knowledge уже существует")
    
    if "chat_history" not in collection_names:
        qdrant_client.create_collection(
            collection_name="chat_history",
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )
        logger.info("Создана коллекция chat_history")
    else:
        logger.info("Коллекция chat_history уже существует")
    
    # Установка вебхуков
    await set_telegram_webhook()
    await set_max_webhook()   # <--- теперь функция определена выше
    
    logger.info("✅ Сервер успешно запущен!")

# ---------- ТОЧКА ВХОДА ----------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
