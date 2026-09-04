import os
import json
import asyncio
import uuid
import random
import traceback
import hashlib
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models


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
    allow_origins=["https://feon1.github.io"],  # Явно указываем ваш домен
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ И НАСТРОЙКИ
# ==========================================
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")
POLZA_API_KEY = os.getenv("POLZA_API_KEY")

# УНИКАЛЬНЫЕ ИМЕНА КОЛЛЕКЦИЙ ДЛЯ TG/WEB
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

def verify_admin(request: Request):
    token = request.headers.get("x-admin-token")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Неверный токен администратора")

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
            print(f"️ Размерность {info.config.params.vectors.size} != 384, пересоздаем...")
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
      ("message_hash", models.PayloadSchemaType.KEYWORD) 
    ]
    for field_name, field_schema in indices:
     try:
        qdrant.create_payload_index(
            collection_name=HISTORY_COLLECTION,
            field_name=field_name,
            field_schema=field_schema
        )
        print(f"✅ Индекс для '{field_name}' создан")
     except Exception as e:
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
        "dimensions": 384
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
            with_payload=True
        )
        if not search_result:
            return ""
        return "\n\n".join([hit.payload.get("text", "") for hit in search_result if hit.payload])
    except Exception as e:
        print(f"⚠️ Ошибка поиска: {e}")
        return ""

async def save_to_history(user_id: str, role: str, content: str):
    """Сохраняет сообщение в историю с надежной защитой от ошибок типов."""
    try:
        # 1. Жесткое приведение всех аргументов к строке
        safe_user_id = str(user_id).strip()
        safe_role = str(role).strip()
        safe_content = str(content).strip() if content is not None else ""
        
        # Нормализация для дедупликации
        normalized_content = ' '.join(safe_content.split())
        
        # Формируем ключ ТОЛЬКО из строк
        content_key = f"{safe_user_id}_{safe_role}_{normalized_content}"
        
        # Передаем СТРОКУ в uuid5 (не bytes!)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, content_key))
        
        # Проверка дубликата
        existing = await asyncio.to_thread(
            qdrant.retrieve,
            collection_name=HISTORY_COLLECTION,
            ids=[point_id],
            with_payload=False
        )
        
        if existing:
            print(f"️ Пропускаем дубликат: {normalized_content[:30]}...")
            return

        # Сохранение
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
                    "timestamp": timestamp
                }
            )]
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
            with_payload=True
        )
        messages = sorted([r.payload for r in records if r.payload], key=lambda x: x.get("timestamp", ""))
        return messages
    except Exception as e:
        print(f"⚠️ Ошибка получения истории: {e}")
        return []

# ==========================================
# 🧠 ЯДРО ЧАТА
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
                {"role": "user", "content": prompt}
            ]
            response = await client.post(
                "https://api.polza.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {POLZA_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "deepseek/deepseek-v4-flash",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 1550
                }
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f" Ошибка Polza API: {e}")
            traceback.print_exc()
            return "Извините, произошла ошибка при обращении к ИИ."
            
    await save_to_history(user_id, "bot", answer)
    return answer

# ==========================================
# 📱 TELEGRAM ИНТЕГРАЦИЯ
# ==========================================
async def send_telegram_message(chat_id: int, text: str):
    if not TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, json=payload)
        except Exception as e:
            print(f"❌ Ошибка отправки в Telegram: {e}")

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
        try:
            response_text = await process_message_core(user_id, text)
            final_text = f"Актуальная версия бота: https://max.ru/se13654625_bot\n\n{response_text}"
            await send_telegram_message(chat_id, final_text)

            #response_text = await process_message_core(user_id, text)
            #await send_telegram_message(chat_id, response_text)
        except Exception as e:
            print(f"❌ Ошибка обработки Telegram: {e}")
            await send_telegram_message(chat_id, "Извините, произошла ошибка.")
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
                payload={"text": text}
            )]
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
            if not para: continue
            if len(current_chunk) + len(para) <= 800:
                current_chunk += (("\n\n" if current_chunk else "") + para)
            else:
                if current_chunk: chunks.append(current_chunk)
                if len(para) > 800:
                    for i in range(0, len(para), 700):
                        chunks.append(para[i:i + 800])
                current_chunk = ""
        if current_chunk: chunks.append(current_chunk)
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
                        payload={"text": chunk, "source_file": file.filename}
                    )]
                )
                success_count += 1
            except Exception as e:
                print(f"⚠️ Пропуск фрагмента {i}: {e}")

        return JSONResponse({"status": "success", "message": f"Добавлено {success_count} из {len(chunks)} фрагментов"})
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

        # Генерируем основной ответ
        base_answer = await process_message_core(user_id, message)
        
        # ---- ВСТАВЬТЕ ПРЕФИКС ----
        prefix = "🤖 Актуальная версия бота: https://feon-chat.website.yandexcloud.net\n\n"
        final_answer = prefix + base_answer
        # -------------------------

        return JSONResponse({"response": final_answer})
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
        qdrant.delete(
            collection_name=HISTORY_COLLECTION,
            points_selector=models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))])
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
                    "length": len(r.payload.get("text", ""))
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
            "files": list(files_stats.values())
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
            points_selector=models.PointIdsList(points=[knowledge_id])
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
            with_payload=False
        )
        if not records:
            return JSONResponse({"error": "Файл не найден"}, status_code=404)
        ids_to_delete = [r.id for r in records]
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(points=ids_to_delete)
        )
        return JSONResponse({
            "status": "success",
            "message": f"Удалено {len(ids_to_delete)} фрагментов из файла {file_name}"
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
            "new_prompt": new_prompt
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
