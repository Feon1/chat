import asyncio
import json
import os
import sys
import websockets
from fastmcp import FastMCP
from fastmcp.tools import Tool
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, Response
from starlette.requests import Request
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# ---- Создаём MCP-сервер и регистрируем инструменты ДО получения http_app ----
mcp = FastMCP("Xiaozhi Direct Adapter")

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

async def send_to_xiaozhi(message: str) -> str:
    print(f"📨 send_to_xiaozhi called with: {message}")
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

            text_msg = {
                "type": "listen",
                "state": "detect",
                "text": message,
                "source": "text"
            }
            await websocket.send(json.dumps(text_msg))
            print("📤 Text message sent")

            full_reply = ""
            while True:
                raw = await websocket.recv()
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
            print(f"✅ Full reply: {full_reply[:100]}...")
            return full_reply if full_reply else "Ответ не получен"

    except Exception as e:
        print(f"❌ Ошибка подключения к Xiaozhi: {e}")
        return f"❌ Ошибка подключения к Xiaozhi: {e}"

# ---- Регистрация инструментов (ДО получения http_app) ----
def send_message_impl(message: str) -> str:
    print(f"🔧 send_message_impl вызван с: {message}")
    return asyncio.run(send_to_xiaozhi(message))

def ping_impl() -> str:
    return "pong"

send_message_tool = Tool.from_function(send_message_impl, name="send_message")
mcp.add_tool(send_message_tool)

ping_tool = Tool.from_function(ping_impl, name="ping")
mcp.add_tool(ping_tool)

print("✅ Инструменты зарегистрированы: send_message, ping")

# ---- Теперь получаем HTTP-приложение ----
app = mcp.http_app()

# ---- Настраиваем CORS и дополнительные маршруты ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["mcp-session-id"],
)

async def options_mcp(request: Request):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Accept, mcp-session-id",
            "Access-Control-Expose-Headers": "mcp-session-id",
        }
    )

app.add_route("/mcp", options_mcp, methods=["OPTIONS"])

async def root(request: Request):
    return JSONResponse({"status": "ok", "service": "Xiaozhi Adapter"})

app.add_route("/", root, methods=["GET", "HEAD"])

print("✅ Инициализация завершена, запускаю сервер...")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
