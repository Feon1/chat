import os
import json
import asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import httpx

# Импорты для RAG (векторный поиск)
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding

# Загрузка переменных окружения
load_dotenv()

app = FastAPI(title="XiaoZhi RAG Adapter")

# ==========================================
# НАСТРОЙКИ RAG (Qdrant + FastEmbed)
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
POLZA_API_KEY = os.getenv("POLZA_API_KEY") # Ваш ключ от Polza.ai / DeepSeek

# Используем легкую модель (весит ~80 МБ, идеально для Render 512MB)
# Поддерживает русский и английский языки
embedding_ = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") 

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
# ФУНКЦИЯ ПОИСКА В БАЗЕ ЗНАНИЙ (RAG)
# ==========================================
async def search_knowledge(query: str) -> str:
    """Ищет релевантный контекст в Qdrant вместо вызова внешнего MCP"""
    try:
        # 1. Получаем векторное представление запроса
        query_embeddings = list(embedding_model.embed([query]))
        query_vector = query_embeddings[0].tolist()
        
        # 2. Ищем топ-3 наиболее похожих фрагмента
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3
        )
        
        if not search_result:
            return ""
            
        # 3. Собираем текст из найденных фрагментов
        fragments = [hit.payload.get("text", "") for hit in search_result if hit.payload]
        context = "\n\n".join(fragments)
        print(f"📚 Найден контекст в Qdrant: {context[:150]}...")
        return context
        
    except Exception as e:
        print(f"⚠️ Ошибка поиска в Qdrant: {e}")
        return ""


# ==========================================
# ЭНДПОИНТ ДЛЯ ДОБАВЛЕНИЯ ЗНАНИЙ
# ==========================================
@app.post("/add_knowledge")
async def add_knowledge(request: Request):
    """Добавляет текстовый фрагмент в векторную базу данных"""
    try:
        body = await request.json()
        text = body.get("text", "")
        
        if not text or len(text.strip()) < 10:
            return JSONResponse({"error": "Текст слишком короткий или отсутствует"}, status_code=400)
            
        # Генерируем вектор для добавляемого текста
        doc_embeddings = list(embedding_model.embed([text]))
        doc_vector = doc_embeddings[0].tolist()
        
        # Сохраняем в Qdrant (ID генерируем на основе хеша текста для уникальности)
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
        return JSONResponse({"status": "success", "message": "Знание успешно добавлено в базу"})
        
    except Exception as e:
        print(f"❌ Ошибка добавления знания: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ==========================================
# ОСНОВНОЙ ЭНДПОИНТ ОБРАБОТКИ ЗАПРОСОВ
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
        # Вы можете изменить это условие под свои нужды
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
                    "https://api.polza.ai/v1/chat/completions", # Уточните актуальный URL Polza.ai
                    headers={
                        "Authorization": f"Bearer {POLZA_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat", # Или другая доступная модель
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3 # Низкая температура для более точных ответов по фактам
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
            
            # ==========================================================
            # ЗДЕСЬ ВАШ СУЩЕСТВУЮЩИЙ КОД ДЛЯ WEBSOCKET XIAOZHI
            # Вставьте сюда вашу рабочую логику отправки коротких запросов
            # ==========================================================
            # Пример заглушки:
            return JSONResponse({
                "answer": "Обработка короткого запроса через WebSocket XiaoZhi (вставьте ваш код здесь)",
                "source": "xiaozhi_short"
            })

    except Exception as e:
        print(f"❌ Критическая ошибка обработки запроса: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ==========================================
# WEBSOCKET ДЛЯ XIAOZHI (Опционально)
# ==========================================
@app.websocket("/ws/xiaozhi")
async def websocket_endpoint(websocket: WebSocket):
    """Эндпоинт для прямого WebSocket соединения, если он вам нужен"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Здесь логика обработки голосового потока или текстовых сообщений
            # await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        print("Клиент отключился от WebSocket")


if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" и port=8000 важны для корректной работы на Render
    uvicorn.run(app, host="0.0.0.0", port=8000)
