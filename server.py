import asyncio
import json
import os
import uuid
import aiohttp
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

# --- Настройки MCP Hub ---
MCP_HUB_URL = os.getenv("MCP_HUB_URL", "https://xiaozhi-mcphub-deploy-server.onrender.com/mcp")
MCP_HUB_TOKEN = os.getenv("MCP_HUB_TOKEN", "")
if not MCP_HUB_TOKEN:
    print("⚠️ MCP_HUB_TOKEN не задан! Поиск в базе знаний недоступен.")
else:
    print("✅ MCP_HUB_TOKEN загружен")

# --- Настройки Polza.ai ---
POLZA_API_KEY = os.getenv("POLZA_API_KEY", "")
POLZA_BASE_URL = "https://polza.ai/api/v1"
POLZA_MODEL = "deepseek/deepseek-v4-flash"

if not POLZA_API_KEY:
    print("⚠️ POLZA_API_KEY не задан! Длинные запросы не будут обрабатываться.")
else:
    print("✅ POLZA_API_KEY загружен")

polza_client = None
if POLZA_API_KEY:
    try:
        from openai import AsyncOpenAI
        polza_client = AsyncOpenAI(api_key=POLZA_API_KEY, base_url=POLZA_BASE_URL)
        print("✅ Клиент Polza.ai создан")
    except Exception as e:
        print(f"⚠️ Ошибка создания клиента Polza.ai: {e}")

# --- Функция вызова MCP Hub с обработкой SSE ---

async def call_mcp_hub(tool_name: str, arguments: dict) -> str:
    if not MCP_HUB_TOKEN:
        return ""

    headers_base = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MCP_HUB_TOKEN}",
        "Accept": "application/json, text/event-stream"
    }

    # 1. Initialize session
    init_payload = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "Xiaozhi Adapter", "version": "1.0.0"}
        },
        "id": 1
    }
    session_id = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(MCP_HUB_URL, headers=headers_base, json=init_payload) as resp:
                if resp.status == 200:
                    session_id = resp.headers.get("mcp-session-id")
                    if not session_id:
                        # try from body
                        data = await resp.json()
                        session_id = data.get("result", {}).get("session_id")
                    if not session_id:
                        print("⚠️ Не удалось получить session_id")
                        return ""
                    print(f"✅ Получен session_id: {session_id}")
                else:
                    error_text = await resp.text()
                    print(f"⚠️ Ошибка инициализации: {resp.status} - {error_text}")
                    return ""
    except Exception as e:
        print(f"⚠️ Ошибка инициализации: {e}")
        return ""

    # 2. Call tool
    headers = headers_base.copy()
    headers["mcp-session-id"] = session_id
    call_payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        },
        "id": 2
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(MCP_HUB_URL, headers=headers, json=call_payload) as resp:
                content_type = resp.headers.get("Content-Type", "")
                if "text/event-stream" in content_type:
                    # Read SSE stream
                    full_text = ""
                    async for line in resp.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                # Extract text from various possible structures
                                if "result" in data:
                                    result = data["result"]
                                    content = result.get("content", [])
                                    for item in content:
                                        if isinstance(item, dict) and "text" in item:
                                            full_text += item["text"]
                                elif "choices" in data:
                                    for choice in data.get("choices", []):
                                        delta = choice.get("delta", {})
                                        if "content" in delta:
                                            full_text += delta["content"]
                                elif "text" in data:
                                    full_text += data["text"]
                            except json.JSONDecodeError:
                                continue
                    if full_text:
                        return full_text
                    else:
                        return ""
                else:
                    # Regular JSON
                    data = await resp.json()
                    result = data.get("result", {})
                    content = result.get("content", [])
                    if content and isinstance(content, list):
                        fragments = [item.get("text", "") for item in content if item.get("text")]
                        if fragments:
                            return "\n\n".join(fragments)
                    structured = result.get("structuredContent", {})
                    if "result" in structured:
                        return structured["result"]
                    return ""
    except Exception as e:
        print(f"⚠️ Ошибка вызова MCP Hub: {e}")
        return ""

# --- Функция для Polza.ai ---

async def call_polza_with_context(prompt: str, context: str) -> str:
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

    if MCP_HUB_TOKEN:
        print("🔍 Выполняем поиск в базе знаний через MCP Hub...")
        context = await call_mcp_hub("search_knowledge", {"query": message})
        if context and not context.startswith("⚠️") and not context.startswith("❌"):
            print(f"📚 Найден контекст: {context[:200]}...")
            return await call_polza_with_context(message, context)
        else:
            # Если контекст пустой или содержит ошибку, не вызываем Polza.ai
            if not context:
                return "❌ Не удалось найти информацию в базе знаний. Пожалуйста, переформулируйте запрос."
            else:
                return context  # Возвращаем сообщение об ошибке от MCP Hub
    else:
        return "⚠️ MCP_HUB_TOKEN не задан! Поиск в базе знаний недоступен."

# --- MCP-обработчик для внешних клиентов ---

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
    return JSONResponse({"status": "ok", "service": "Xiaozhi Adapter (MCP Hub + Polza.ai)"})

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
                    "serverInfo": {"name": "Xiaozhi Adapter", "version": "1.0.0"}
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
