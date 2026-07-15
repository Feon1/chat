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

app = FastAPI(title="XiaoZhi RAG Adapter (Lightweight)")

# ==========================================
# НАСТРОЙКИ ИЗ ПЕРЕМЕННЫХ ОКРУЖЕНИЯ
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
POLZA_API_KEY = os.getenv("POLZA_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN", "") # Ваш токен Hugging Face

COLLECTION_NAME = "xiaozhi_knowledge"

# URL бесплатного API Hugging Face для мультиязычных эмбеддингов
EMBEDDING_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

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
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: ПОЛУЧЕНИЕ ВЕКТОРА
# ==========================================
async def get_embedding(text: str) -> list[float]:
    """Получает вектор текста через API Hugging Face с надежной обработкой ошибок"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            EMBEDDING_API_URL,
            json={"inputs": text, "options": {"wait_for_model": True}},
            headers=headers,
            timeout=30.0
        )
        
        # 1. Проверяем код ответа
        if response.status_code != 200:
            error_msg = f"HF API вернул ошибку {response.status_code}: {response.text}"
            print(f"❌ ОШИБКА HF API: {error_msg}")
            raise Exception(error_msg)
            
        result = response.json()
        
        # 2. Проверяем, что пришел именно список чисел, а не сообщение об ошибке в JSON
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
            return result[0]
        else:
            error_msg = f"Неожиданный формат ответа от HF: {result}"
            print(f"❌ ОШИБКА ФОРМАТА HF: {error_msg}")
            raise Exception(error_msg)


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
            
        print(f"🔄 Запрос вектора для текста: '{text[:50]}...'")
        doc_vector = await get_embedding(text)
        print("✅ Вектор успешно получен от Hugging Face")
        
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=abs(hash(text)) % 1000000000, # Уникальный ID на основе хеша
                    vector=doc_vector,
                    payload={"text": text}
                )
            ]
        )
        print("✅ Успешно сохранено в Qdrant")
        return JSONResponse({"status": "success", "message": "Знание успешно добавлено в базу"})
        
    except Exception as e:
        # Эта строка выведет точную причину в логи Render!
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
            limit=3 # Берем топ-3 наиболее похожих фрагмента
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

        # Логика разделения: запросы длиннее 40 символов считаем "сложными"
        if len(text) > 40:
            print(f"🧠 Длинный запрос, используем RAG + LLM: '{text}'")
            
            # 1. Ищем контекст в нашей базе знаний
            context = await search_knowledge(text)
            
            # 2. Формируем промпт для LLM
            if context:
                prompt = (
                    "Ты умный помощник. Используй следующий КОНТЕКСТ для ответа на вопрос. "
                    "Если в контексте нет точного ответа, используй свои общие знания, но отдавай приоритет контексту.\n\n"
                    f"КОНТЕКСТ:\n{context}\n\n"
                    f"ВОПРОС ПОЛЬЗОВАТЕЛЯ: {text}"
                )
            else:
                prompt = f"ВОПРОС ПОЛЬЗОВАТЕЛЯ: {text}"
            
            # 3. Вызываем внешний LLM (DeepSeek через Polza.ai)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.polza.ai/v1/chat/completions", # Уточните актуальный URL Polza.ai, если он другой
                    headers={
                        "Authorization": f"Bearer {POLZA_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat", # Или другая доступная модель
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3 # Низкая температура для точных ответов
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
            # СЮДА ВСТАВЛЯЕТСЯ ВАШ КОД ДЛЯ WEBSOCKET XIAOZHI
            return JSONResponse({
                "answer": "Обработка короткого запроса через WebSocket XiaoZhi (вставьте ваш код здесь)",
                "source": "xiaozhi_short"
            })

    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА в /query: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    # Render использует порт 10000 по умолчанию
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
