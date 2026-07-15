import asyncio
import json
import os
import uuid
import websockets
from openai import AsyncOpenAI
from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, Response
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["mcp-session-id"],
)

sessions = {}

# --- Настройки Xiaozhi MCP ---
XIAOZHI_MCP_URL = os.getenv("XIAOZHI_MCP_URL", "wss://api.xiaozhi.me/mcp/")
XIAOZHI_MCP_TOKEN = os.getenv("XIAOZHI_MCP_TOKEN", "")
if not XIAOZHI_MCP_TOKEN:
    print("⚠️  XIAOZHI_MCP_TOKEN не задан! Поиск в базе знаний будет недоступен.")
else:
    print("✅ XIAOZHI_MCP_TOKEN загружен")

# --- Настройки Polza.ai ---
POLZA_API_KEY = os.getenv("POLZA_API_KEY", "")
POLZA_BASE_URL = "https://polza.ai/api/v1"
POLZA_MODEL = "deepseek/deepseek-v4-flash"

if not POLZA_API_KEY:
    print("⚠️  POLZA_API_KEY не задан! Длинные запросы не будут обрабатываться.")
else:
    print("✅ POLZA_API_KEY загружен")

polza_client = None
if POLZA_API_KEY:
    polza_client = AsyncOpenAI(
        api_key=POLZA_API_KEY,
        base_url=POLZA_BASE_URL,
    )

# --- Вспомогательные функции ---

async def call_mcp_search_knowledge(query: str) -> str:
    """
    Подключается к MCP-эндпоинту Xiaozhi, выполняет инициализацию,
    вызывает search_knowledge и возвращает контекст.
    """
    if not XIAOZHI_MCP_TOKEN:
        print("⚠️ XIAOZHI_MCP_TOKEN отсутствует")
        return ""

    ws_url = f"{XIAOZHI_MCP_URL}?token={XIAOZHI_MCP_TOKEN}"
    print(f"🔗 Подключение к Xiaozhi MCP WebSocket: {ws_url[:80]}...")

    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket подключен к Xiaozhi MCP")

            # Шаг 1: Отправить initialize
            init_msg = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "Xiaozhi Adapter", "version": "1.0.0"}
                },
                "id": 1
            }
            await websocket.send(json.dumps(init_msg))
            print("📤 Отправлен initialize")

            # Ждём ответ на initialize
            try:
                resp = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                data = json.loads(resp)
                print(f"📩 Получен ответ на initialize: {data.get('id')}")
                if data.get("id") != 1:
                    print("⚠️ Неверный ID в ответе на initialize")
                    return ""
            except asyncio.TimeoutError:
                print("⏰ Таймаут при initialize")
                return ""
            except Exception as e:
                print(f"⚠️ Ошибка при initialize: {e}")
                return ""

            # Шаг 2: Отправить notifications/initialized
            notify_msg = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }
            await websocket.send(json.dumps(notify_msg))
            print("📤 Отправлен notifications/initialized")

            # Шаг 3: Вызвать search_knowledge
            call_msg = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "search_knowledge",
                    "arguments": {"query": query}
                },
                "id": 2
            }
            await websocket.send(json.dumps(call_msg))
            print("📤 Вызов search_knowledge отправлен")

            # Шаг 4: Читаем ответы, собираем контекст
            while True:
                try:
                    resp = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                except asyncio.TimeoutError:
                    print("⏰ Таймаут ожидания ответа от search_knowledge")
                    break
                try:
                    data = json.loads(resp)
                    print(f"📩 Получено сообщение: {data.get('id')}")
                except json.JSONDecodeError:
                    continue

                # Проверяем, что это ответ на tools/call
                if data.get("id") == 2:
                    if "error" in data:
                        error_msg = data["error"].get("message", "Неизвестная ошибка")
                        print(f"❌ Ошибка от Xiaozhi: {error_msg}")
                        return ""
                    result = data.get("result", {})
                    content = result.get("content", [])
                    fragments = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            fragments.append(item["text"])
                    if fragments:
                        context = "\n\n".join(fragments)
                        print(f"📚 Найден контекст: {context[:200]}...")
                        return context
                    else:
                        return ""
                else:
                    # Игнорируем другие сообщения (например, уведомления)
                    continue
            return ""
    except Exception as e:
        print(f"⚠️ Ошибка вызова search_knowledge: {e}")
        import traceback
        traceback.print_exc()
        return ""

async def call_polza_with_context(prompt: str, context: str) -> str:
    """Отправляет запрос в Polza.ai с контекстом."""
    if not POLZA_API_KEY:
        return "⚠️ Polza.ai не настроен. Установите POLZA_API_KEY."

    if not polza_client:
        return "⚠️ Клиент Polza.ai недоступен."

    if not context or not context.strip():
        return "❌ Не удалось найти информацию в базе знаний. Пожалуйста, переформулируйте запрос."

    system_prompt = "Ты — полезный ассистент. Отвечай на вопрос, используя предоставленный контекст."
    system_prompt += f"\n\nКонтекст:\n{context}"

    try:
        response = await polza_client.chat.completions.create(
            model=POLZA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=2000,
        )
        return response.choices[0].message.content or "Ответ не получен"
    except Exception as e:
        return f"⚠️ Ошибка при вызове Polza.ai: {e}"

# --- Основная функция обработки запросов ---

async def send_to_xiaozhi(message: str) -> str:
    print(f"📨 send_to_xiaozhi called with: {message}")

    # Для любых запросов (любой длины) – сначала ищем контекст в базе знаний Xiaozhi
    if XIAOZHI_MCP_TOKEN:
        print("🔍 Выполняем поиск в базе знаний через Xiaozhi MCP...")
        context = await call_mcp_search_knowledge(message)
        if context:
            print("✅ Контекст получен, отправляем в Polza.ai")
            return await call_polza_with_context(message, context)
        else:
            return "❌ Не удалось найти информацию в базе знаний. Пожалуйста, переформулируйте запрос."
    else:
        return "⚠️ XIAOZHI_MCP_TOKEN не задан! Поиск в базе знаний недоступен."

# --- MCP-обработчик ---

@app.options("/mcp")
async def options_mcp():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Accept, mcp-session-id",
            "Access-Control-Expose-Headers": "mcp-session-id",
        }
    )

@app.get("/")
async def root():
    return JSONResponse({"status": "ok", "service": "Xiaozhi Adapter (RAG + Polza.ai)"})

@app.post("/mcp")
async def mcp_handler(request: Request):
    try:
        body = await request.json()
        print(f"📩 POST /mcp body: {body}")
        method = body.get("method")
        session_id = request.headers.get("mcp-session-id")

        if method == "initialize":
            new_session_id = str(uuid.uuid4()).replace("-", "")
            sessions[new_session_id] = {"active": True}
            response_data = {
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "Xiaozhi Adapter (RAG)", "version": "1.0.0"}
                }
            }
            response = JSONResponse(response_data)
            response.headers["mcp-session-id"] = new_session_id
            return response

        if method == "notifications/initialized":
            return Response(status_code=200)

        if not session_id or session_id not in sessions:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {"code": -32000, "message": "Bad Request: No valid session ID provided"}
            }, status_code=400)

        if method == "tools/call":
            params = body.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if tool_name == "send_message":
                message = arguments.get("message", "")
                result_text = await send_to_xiaozhi(message)
                sse_data = {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "content": [{"type": "text", "text": result_text}],
                        "structuredContent": {"result": result_text}
                    }
                }
                sse_body = f"event: message\ndata: {json.dumps(sse_data)}\n\n"
                return Response(content=sse_body, media_type="text/event-stream")

            else:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {"code": -32602, "message": f"Unknown tool: {tool_name}"}
                }, status_code=400)

        return JSONResponse({
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        }, status_code=400)

    except Exception as e:
        print(f"❌ Ошибка в mcp_handler: {e}")
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": body.get("id") if 'body' in locals() else None,
            "error": {"code": -32603, "message": str(e)}
        }, status_code=500)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
