import asyncio
import json
import os
import websockets
from fastmcp import FastMCP
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("Xiaozhi Direct Adapter")
app = mcp.http_app()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["mcp-session-id"],
)

# ---- Корневой маршрут для health check ----
async def root(request):
    return JSONResponse({"status": "ok", "service": "Xiaozhi Adapter"})

app.add_route("/", root, methods=["GET", "HEAD"])

print(f"🔍 websockets version: {websockets.__version__}")

XIAOZHI_WS_URL = os.getenv("XIAOZHI_WS_URL", "wss://api.tenclass.net/xiaozhi/v1/")
XIAOZHI_TOKEN = os.getenv("XIAOZHI_TOKEN", "")
if not XIAOZHI_TOKEN:
    print("⚠️  ВНИМАНИЕ: XIAOZHI_TOKEN не задан!")

DEVICE_ID = os.getenv("DEVICE_ID", "e0:2e:0b:ae:79:ea")
CLIENT_ID = os.getenv("CLIENT_ID", "9cc3e5e4-adcf-4eff-8d23-95d4eaa21020")

async def send_to_xiaozhi(message: str) -> str:
    headers = {
        "Device-Id": DEVICE_ID,
        "Client-Id": CLIENT_ID,
        "Protocol-Version": "1",
    }
    ws_url = f"{XIAOZHI_WS_URL}?token={XIAOZHI_TOKEN}"

    try:
        # Универсальный способ: преобразуем словарь в список кортежей
        headers_list = list(headers.items())
        # Для некоторых версий нужно передавать extra_headers именно как список кортежей
        async with websockets.connect(ws_url, extra_headers=headers_list) as websocket:
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

            try:
                resp = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(resp)
                if data.get("type") != "hello":
                    return f"Ошибка: ожидался hello, получено {data.get('type')}"
                session_id = data.get("session_id")
                if not session_id:
                    return "Ошибка: не получен session_id"
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

            full_reply = ""
            while True:
                raw = await websocket.recv()
                if isinstance(raw, bytes):
                    continue
                try:
                    data = json.loads(raw)
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
            return full_reply if full_reply else "Ответ не получен"

    except Exception as e:
        return f"❌ Ошибка подключения к Xiaozhi: {e}"

@mcp.tool()
def send_message(message: str) -> str:
    return asyncio.run(send_to_xiaozhi(message))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
