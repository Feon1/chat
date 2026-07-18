import os
import json
import asyncio
import uuid
from datetime import datetime
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

app = FastAPI(title="XiaoZhi RAG Adapter")

# Разрешаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# НАСТРОЙКИ
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
POLZA_API_KEY = os.getenv("POLZA_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

COLLECTION_NAME = "xiaozhi_knowledge"
HISTORY_COLLECTION = "chat_history"
JINA_API_URL = "https://api.jina.ai/v1/embeddings"

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@app.on_event("startup")
async def startup_event():
    """Создаем коллекции при запуске"""
    # Коллекция для базы знаний
    try:
        qdrant.get_collection(COLLECTION_NAME)
        print(f"✅ Коллекция '{COLLECTION_NAME}' найдена")
    except Exception:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )
        print(f"✅ Коллекция '{COLLECTION_NAME}' создана")
    
    # Коллекция для истории чатов (используем фиктивный вектор размера 1)
    try:
        qdrant.get_collection(HISTORY_COLLECTION)
        print(f"✅ Коллекция '{HISTORY_COLLECTION}' найдена")
    except Exception:
        qdrant.create_collection(
            collection_name=HISTORY_COLLECTION,
            vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
        )
        print(f"✅ Коллекция '{HISTORY_COLLECTION}' создана")

# ==========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
async def get_embedding(text: str) -> list[float]:
    """Получает вектор через Jina AI"""
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        response = await client.post(
            JINA_API_URL,
            json={"model": "jina-embeddings-v3", "input": [text], "task": "text-matching", "dimensions": 384},
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

async def search_knowledge(query: str) -> str:
    """Ищет контекст в базе знаний"""
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

def save_to_history(user_id: str, role: str, content: str):
    """Сохраняет сообщение в историю (Qdrant)"""
    try:
        message_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Используем фиктивный вектор [1.0], так как поиск по истории не нужен
        qdrant.upsert(
            collection_name=HISTORY_COLLECTION,
            points=[
                models.PointStruct(
                    id=abs(hash(message_id)) % 1000000000,
                    vector=[1.0],
                    payload={
                        "message_id": message_id,
                        "user_id": user_id,
                        "role": role,
                        "content": content,
                        "timestamp": timestamp
                    }
                )
            ]
        )
        print(f"✅ Сообщение сохранено в историю для {user_id}")
    except Exception as e:
        print(f"⚠️ Ошибка сохранения истории: {e}")

def get_history(user_id: str, limit: int = 50) -> list[dict]:
    """Получает историю сообщений пользователя"""
    try:
        # Используем scroll для получения всех сообщений пользователя
        records, next_page = qdrant.scroll(
            collection_name=HISTORY_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            ),
            limit=limit,
            with_payload=True
        )
        
        # Сортируем по времени (от старых к новым)
        messages = sorted(
            [r.payload for r in records if r.payload],
            key=lambda x: x.get("timestamp", "")
        )
        
        return messages
    except Exception as e:
        print(f"⚠️ Ошибка получения истории: {e}")
        return []

# ==========================================
# ЭНДПОИНТЫ
# ==========================================
@app.get("/")
def read_root():
    """Проверка работоспособности"""
    return {"status": "running", "message": "XiaoZhi RAG Adapter работает!"}

@app.post("/add_knowledge")
async def add_knowledge(request: Request):
    """Добавляет текст в базу знаний"""
    try:
        body = await request.json()
        text = body.get("text", "")
        if not text or len(text.strip()) < 10:
            return JSONResponse({"error": "Текст слишком короткий"}, status_code=400)
        
        doc_vector = await get_embedding(text)
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[models.PointStruct(
                id=abs(hash(text)) % 1000000000,
                vector=doc_vector,
                payload={"text": text}
            )]
        )
        return JSONResponse({"status": "success", "message": "Знание добавлено"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    """Загружает PDF/DOCX в базу знаний"""
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
        
        # Разбиваем на чанки
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
                else:
                    current_chunk = para
        if current_chunk:
            chunks.append(current_chunk)
        chunks = [c for c in chunks if len(c.strip()) > 30]
        
        success_count = 0
        for i, chunk in enumerate(chunks):
            try:
                doc_vector = await get_embedding(chunk)
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[models.PointStruct(
                        id=abs(hash(f"{file.filename}_{i}")) % 1000000000,
                        vector=doc_vector,
                        payload={"text": chunk, "source_file": file.filename}
                    )]
                )
                success_count += 1
            except Exception as e:
                print(f"⚠️ Пропуск фрагмента {i}: {e}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Добавлено {success_count} из {len(chunks)} фрагментов"
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/query")
async def handle_query(request: Request):
    """Обработка запроса с историей"""
    try:
        body = await request.json()
        text = body.get("text", "")
        user_id = body.get("user_id", "anonymous")
        
        if not text:
            return JSONResponse({"error": "Текст пуст"}, status_code=400)

        # Сохраняем сообщение пользователя
        save_to_history(user_id, "user", text)

        if len(text) > 40:
            print(f" Длинный запрос от {user_id}")
            
            # Получаем последние 5 сообщений для контекста
            history = get_history(user_id, limit=10)
            recent_history = history[-5:] if len(history) > 5 else history
            
            chat_history_str = ""
            for msg in recent_history:
                role = "Пользователь" if msg['role'] == 'user' else "Ассистент"
                chat_history_str += f"{role}: {msg['content']}\n"

            context = await search_knowledge(text)
            
            prompt = ""
            if chat_history_str:
                prompt += f"История диалога:\n{chat_history_str}\n\n"
            if context:
                prompt += f"Контекст из базы знаний:\n{context}\n\n"
            prompt += f"Вопрос: {text}"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.polza.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {POLZA_API_KEY}", "Content-Type": "application/json"},
                    json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3},
                    timeout=30.0
                )
                answer = response.json()["choices"][0]["message"]["content"]
            
            # Сохраняем ответ бота
            save_to_history(user_id, "bot", answer)
            
            return JSONResponse({"answer": answer, "source": "rag_llm"})
        else:
            return JSONResponse({"answer": "Короткий запрос", "source": "short"})

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/get_history")
async def get_history_endpoint(user_id: str):
    """Возвращает историю пользователя"""
    try:
        messages = get_history(user_id, limit=50)
        return JSONResponse({"messages": messages})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
