import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Загрузка переменных окружения
load_dotenv()

app = FastAPI(title="XiaoZhi RAG Adapter (Jina)")

# ==========================================
# НАСТРОЙКИ ИЗ ПЕРЕМЕННЫХ ОКРУЖЕНИЯ
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
POLZA_API_KEY = os.getenv("POLZA_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY") # Новый ключ Jina

COLLECTION_NAME = "xiaozhi_knowledge"

# Инициализация клиента Qdrant
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@app.on_event("startup")
async def startup_event():
    """Проверяем или создаем коллекцию при запуске сервера"""
    try:
        qdrant.get_collection(COLLECTION_NAME)
        print(f"✅ Коллекция '{COLLECTION_NAME}' успешно найдена в Qdrant")
    except Exception:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )
        print(f"✅ Коллекция '{COLLECTION_NAME}' успешно создана в Qdrant")


# ==========================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: ПОЛУЧЕНИЕ ВЕКТОРА (JINA)
# ==========================================
async def get_embedding(text: str) -> list[float]:
    """Получает вектор текста через стабильный API Jina AI"""
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.jina.ai/v1/embeddings",
            json={
                "model": "jina-embeddings-v3", # Отличная мультиязычная модель
                "input": [text],
                "task": "text-matching",
                "dimensions": 384 # Запрашиваем 384, чтобы совпадало с Qdrant!
            },
            headers=headers,
            timeout=30.0
        )
        
        # Jina возвращает четкую ошибку, если что-то не так
        response.raise_for_status()
        result = response.json()
        
        # Извлекаем вектор из ответа Jina
        return result["data"][0]["embedding"]


# ==========================================
# ЭНДПОИНТ 1: ДОБАВЛЕНИЕ ЗНАНИЙ
# ==========================================
@app.post("/add_knowledge")
async def add_knowledge(request: Request):
    """Добавляет текстовый фрагмент в векторную базу данных"""
    try:
        body = await request.json()
        text = body.get("text", "")
        
        if not text or len(text.strip()) < 10:
            return JSONResponse({"error": "Текст слишком короткий или отсутствует"}, status_code=400)
            
        print(f"🔄 Запрос вектора Jina для текста: '{text[:50]}...'")
        doc_vector = await get_embedding(text)
        print("✅ Вектор успешно получен от Jina AI")
        
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=abs(hash(text)) % 1000000000,
                    vector=doc_vector,
                    payload={"text": text}
                )
            ]
        )
        print("✅ Успешно сохранено в Qdrant")
        return JSONResponse({"status": "success", "message": "Знание успешно добавлено в базу"})
        
    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА в /add_knowledge: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ==========================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: ПОИСК В БАЗЕ
# ==========================================
async def search_knowledge(query: str) -> str:
    """Ищет релевантный контекст в Qdrant"""
    try:
        query_vector = await get_embedding(query)
        
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3
        )
        
        if not search_result:
            return ""
            
        fragments = [hit.payload.get("text", "") for hit in search_result if hit.payload]
        return "\n\n".join(fragments)
        
    except Exception as e:
        print(f"⚠️ Ошибка поиска в Qdrant: {e}")
        return ""


# ==========================================
# ЭНДПОИНТ 2: ОБРАБОТКА ЗАПРОСА (МАРШРУТИЗАТОР)
# ==========================================
@app.post("/query")
async def handle_query(request: Request):
    """Маршрутизатор: короткие запросы -> XiaoZhi, длинные -> RAG + LLM"""
    try:
        body = await request.json()
        text = body.get("text", "")
        
        if not text:
            return JSONResponse({"error": "Текст запроса пуст"}, status_code=400)

        if len(text) > 40:
            print(f"🧠 Длинный запрос, используем RAG + LLM: '{text}'")
            
            context = await search_knowledge(text)
            
            if context:
                prompt = (
                    "Ты умный помощник. Используй следующий КОНТЕКСТ для ответа на вопрос. "
                    "Если в контексте нет точного ответа, используй свои общие знания, но отдавай приоритет контексту.\n\n"
                    f"КОНТЕКСТ:\n{context}\n\n"
                    f"ВОПРОС ПОЛЬЗОВАТЕЛЯ: {text}"
                )
            else:
                prompt = f"ВОПРОС ПОЛЬЗОВАТЕЛЯ: {text}"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.polza.ai/v1/chat/completions", 
                    headers={
                        "Authorization": f"Bearer {POLZA_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                
            return JSONResponse({
                "answer": answer, 
                "source": "rag_llm",
                "context_used": bool(context)
            })
        
        else:
            print(f"⚡ Короткий запрос, перенаправляем в XiaoZhi: '{text}'")
            return JSONResponse({
                "answer": "Обработка короткого запроса через WebSocket XiaoZhi (вставьте ваш код здесь)",
                "source": "xiaozhi_short"
            })

    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА в /query: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
