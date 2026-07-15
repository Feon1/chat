import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

app = FastAPI(title="XiaoZhi RAG Adapter (Lightweight)")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
POLZA_API_KEY = os.getenv("POLZA_API_KEY")
# Токен HF необязателен для редких запросов, но с ним лимиты выше. Можно получить бесплатно на huggingface.co/settings/tokens
HF_TOKEN = os.getenv("HF_TOKEN", "") 

COLLECTION_NAME = "xiaozhi_knowledge"
# Используем ту же легкую мультиязычную модель, но через API
EMBEDDING_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@app.on_event("startup")
async def startup_event():
    try:
        qdrant.get_collection(COLLECTION_NAME)
        print(f"✅ Коллекция '{COLLECTION_NAME}' найдена в Qdrant")
    except Exception:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )
        print(f"✅ Коллекция '{COLLECTION_NAME}' создана в Qdrant")

async def get_embedding(text: str) -> list[float]:
    """Получает вектор текста через бесплатный API Hugging Face (0 МБ локальной памяти!)"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    async with httpx.AsyncClient() as client:
        response = await client.post(
            EMBEDDING_API_URL,
            json={"inputs": text, "options": {"wait_for_model": True}},
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        # HF API возвращает список списков, берем первый элемент
        return response.json()[0]

async def search_knowledge(query: str) -> str:
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

@app.post("/add_knowledge")
async def add_knowledge(request: Request):
    try:
        body = await request.json()
        text = body.get("text", "")
        if not text or len(text.strip()) < 10:
            return JSONResponse({"error": "Текст слишком короткий"}, status_code=400)
            
        doc_vector = await get_embedding(text)
        
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
        return JSONResponse({"status": "success", "message": "Знание добавлено"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/query")
async def handle_query(request: Request):
    try:
        body = await request.json()
        text = body.get("text", "")
        if not text:
            return JSONResponse({"error": "Текст запроса пуст"}, status_code=400)

        if len(text) > 40:
            print(f"🧠 Длинный запрос, используем RAG: '{text}'")
            context = await search_knowledge(text)
            
            if context:
                prompt = (
                    "Ты умный помощник. Используй КОНТЕКСТ для ответа. Если в нем нет ответа, используй общие знания.\n\n"
                    f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {text}"
                )
            else:
                prompt = f"ВОПРОС: {text}"
            
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
            print(f"⚡ Короткий запрос: '{text}'")
            return JSONResponse({
                "answer": "Здесь будет ваш код для коротких запросов XiaoZhi",
                "source": "xiaozhi_short"
            })
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000) # Render использует порт 10000
