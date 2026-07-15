import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
import io
from pypdf import PdfReader
from docx import Document

# Загрузка переменных окружения
load_dotenv()

app = FastAPI(title="XiaoZhi RAG Adapter (Jina)")
# Разрешаем запросы с вашего GitHub Pages и любых других источников
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене лучше указать ["https://feon1.github.io"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Разбивает длинный текст на перекрывающиеся фрагменты для лучшего поиска"""
    chunks = []
    # Сначала разбиваем на абзацы, чтобы не резать предложения посередине
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += (("\n\n" if current_chunk else "") + para)
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Если один абзац сам по себе больше chunk_size, режем его жестко
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i:i + chunk_size])
                current_chunk = ""
            else:
                current_chunk = para
                
    if current_chunk:
        chunks.append(current_chunk)
        
    # Фильтруем слишком короткие бессмысленные куски
    return [c for c in chunks if len(c.strip()) > 30]
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

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    """Загружает PDF или DOCX, разбивает на чанки и добавляет в Qdrant"""
    try:
        filename = file.filename.lower()
        content = await file.read()
        text = ""
        
        print(f"📄 Обработка файла: {file.filename} ({len(content)} байт)")
        
        if filename.endswith('.pdf'):
            reader = PdfReader(io.BytesIO(content))
            text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
        elif filename.endswith('.docx'):
            doc = Document(io.BytesIO(content))
            text = "\n\n".join([para.text for para in doc.paragraphs])
        else:
            return JSONResponse({"error": "Поддерживаются только .pdf и .docx"}, status_code=400)
            
        if not text.strip():
            return JSONResponse({"error": "Не удалось извлечь текст из документа"}, status_code=400)
            
        # Разбиваем на чанки
        chunks = chunk_text(text)
        print(f"✅ Документ разбит на {len(chunks)} фрагментов")
        
        success_count = 0
        for i, chunk in enumerate(chunks):
            try:
                # Получаем вектор для каждого фрагмента
                doc_vector = await get_embedding(chunk)
                
                # Сохраняем в Qdrant, добавляя имя файла в payload для справки
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        models.PointStruct(
                            id=abs(hash(f"{file.filename}_{i}")) % 1000000000,
                            vector=doc_vector,
                            payload={"text": chunk, "source_file": file.filename}
                        )
                    ]
                )
                success_count += 1
                if i % 5 == 0:
                    print(f"  ⏳ Обработано {i}/{len(chunks)} фрагментов...")
            except Exception as e:
                print(f"  ⚠️ Пропуск фрагмента {i}: {e}")
                
        return JSONResponse({
            "status": "success", 
            "message": f"Успешно добавлено {success_count} из {len(chunks)} фрагментов из файла {file.filename}"
        })
        
    except Exception as e:
        print(f"❌ Ошибка загрузки документа: {str(e)}")
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
            limit=3,
            with_payload=True  # Явно указываем, что нам нужен текст (payload)
        )
        
        if not search_result:
            return ""
            
        fragments = [hit.payload.get("text", "") for hit in search_result if hit.payload]
        return "\n\n".join(fragments)
        
    except AttributeError as e:
        print(f"⚠️ Ошибка версии Qdrant (AttributeError): {e}")
        return ""
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

        if len(text) > 1:
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
                        "model": "deepseek/deepseek-v4-flash",
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
