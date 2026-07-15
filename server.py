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

# --- Настройки Xiaozhi ---
XIAOZHI_WS_URL = os.getenv("XIAOZHI_WS_URL", "wss://api.tenclass.net/xiaozhi/v1/")
XIAOZHI_TOKEN = os.getenv("XIAOZHI_TOKEN", "")
if not XIAOZHI_TOKEN:
    print("⚠️  XIAOZHI_TOKEN не задан!")
else:
    print(f"✅ XIAOZHI_TOKEN загружен: {XIAOZHI_TOKEN[:10]}...")

DEVICE_ID = os.getenv("DEVICE_ID", "e0:2e:0b:ae:79:ea")
CLIENT_ID = os.getenv("CLIENT_ID", "9cc3e5e4-adcf-4eff-8d23-95d4eaa21020")
print(f"📱 Device ID: {DEVICE_ID}")
print(f"📱 Client ID: {CLIENT_ID}")

# --- Настройки Polza.ai ---
POLZA_API_KEY = os.getenv("POLZA_API_KEY", "")
POLZA_BASE_URL = "https://polza.ai/api/v1"          # OpenAI-совместимый эндпоинт Polza
POLZA_MODEL = "deepseek/deepseek-r1-distill-llama-70b"  # Модель через Polza

if not POLZA_API_KEY:
    print("⚠️  POLZA_API_KEY не задан! Длинные запросы не будут обрабатываться.")

# Инициализация клиента Polza.ai (совместим с OpenAI)
polza_client = AsyncOpenAI(
    api_key=POLZA_API_KEY,
    base_url=POLZA_BASE_URL,
)

# --- Вспомогательные функции ---

async def call_polza(prompt: str) -> str:
    """Вызов модели DeepSeek через Polza.ai с белым списком провайдера Chutes."""
    if not POLZA_API_KEY:
        return "⚠️ Polza.ai не настроен. Установите POLZA_API_KEY в переменных окружения."

    try:
        # Используем extra_body для передачи параметра provider с белым списком
        response = await polza_client.chat.completions.create(
            model=POLZA_MODEL,
            messages=[
                {"role": "system", "content": "Ты — полезный и дружелюбный ассистент."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=2000,
            extra_body={
                "provider": {
                    "only": ["Chutes"]  # Белый список — использовать только Chutes
                }
            }
        )
        return response.choices[0].message.content or "Ответ не получен"
    except Exception as e:
        return f"⚠️ Ошибка при вызове Polza.ai: {e}"

async def send_to_xiaozhi(message: str) -> str:
    print(f"📨 send_to_xiaozhi called with: {message}")

    # Если сообщение длиннее 50 символов — отправляем в Polza.ai (Chutes)
    if len(message) > 1:
        return await call_polza(message)

    # --- Короткие запросы (≤50 символов) — через WebSocket Xiaozhi ---
    headers = {
        "Device-Id": DEVICE_ID,
        "Client-Id": CLIENT_ID,
        "Protocol-Version": "1",
    }
    ws_url = f"{XIAOZHI_WS_URL}?token={XIAOZHI_TOKEN}"
    print(f"🔗 Connecting to: {ws_url[:60]}...")

    try:
        async with websockets.connect(ws_url, extra_headers=headers) as websocket:
            print("✅ WebSocket connected to Xiaozhi")
            hello = {
                "type": "hello",
                "version": 1,
                "transport": "websocket",
                "audio_params": {
                    "format": "opus",
                    "sample_rate": 16000,
                    "channels": 1,
                    "frame_duration": 60
                }
            }
            await websocket.send(json.dumps(hello))
            print("📤 Hello sent")

            try:
                resp = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"📩 Received: {resp[:100]}...")
                data = json.loads(resp)
                if data.get("type") != "hello":
                    return f"Ошибка: ожидался hello, получено {data.get('type')}"
                session_id = data.get("session_id")
                if not session_id:
                    return "Ошибка: не получен session_id"
                print(f"✅ Получен session_id: {session_id}")
            except asyncio.TimeoutError:
                return "⏰ Таймаут: сервер не ответил на hello"
            except Exception as e:
                return f"❌ Ошибка при получении hello: {e}"

            text_msg = {"type": "listen", "state": "detect", "text": message, "source": "text"}
            await websocket.send(json.dumps(text_msg))
            print("📤 Отправлен detect")

            full_reply = ""
            while True:
                try:
                    raw = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                except asyncio.TimeoutError:
                    if full_reply:
                        print("⏰ Таймаут, но есть ответ, возвращаем накопленный текст")
                        return full_reply
                    else:
                        return "⏰ Таймаут ожидания ответа от Xiaozhi"
                if isinstance(raw, bytes):
                    print("📩 Бинарные данные (аудио) пропущены")
                    continue
                try:
                    data = json.loads(raw)
                    print(f"📩 JSON: {data}")
                except json.JSONDecodeError:
                    continue
                msg_type = data.get("type")
                if msg_type == "stt":
                    continue
                elif msg_type == "llm":
                    if "text" in data and data["text"].strip():
                        full_reply += data["text"]
                elif msg_type == "tts":
                    if data.get("state") == "sentence_start":
                        if "text" in data and data["text"].strip():
                            full_reply += data["text"]
                    elif data.get("state") in ("end", "stop"):
                        break
                elif msg_type == "error":
                    return f"Ошибка от Xiaozhi: {data.get('message', 'неизвестная')}"
                elif msg_type == "alert":
                    return f"Ошибка Xiaozhi: {data.get('message', 'неизвестная')}"
                else:
                    print(f"⚠️ Неизвестный тип сообщения: {msg_type}")

            print(f"✅ Full reply: {full_reply[:100]}...")
            await websocket.close()
            return full_reply if full_reply else "Ответ не получен"

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"❌ Соединение закрыто аварийно: {e}")
        return "❌ Ошибка соединения с Xiaozhi"
    except Exception as e:
        print(f"❌ Ошибка подключения к Xiaozhi: {e}")
        return f"❌ Ошибка подключения к Xiaozhi: {e}"

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
    return JSONResponse({"status": "ok", "service": "Xiaozhi Adapter (Polza.ai + Chutes)"})

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
                    "serverInfo": {"name": "Xiaozhi Adapter (Polza.ai + Chutes)", "version": "1.0.0"}
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
