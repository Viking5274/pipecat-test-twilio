import asyncio
import os
import sys
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from starlette.responses import HTMLResponse

from bot_prototype import run_bot

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/start_call')
async def start_call():
    print("POST TwiML")
    return HTMLResponse(content=open("templates/streams.xml").read(), media_type="application/xml")


def start_bot(websocket):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_bot(websocket))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    bot_thread = threading.Thread(target=start_bot, args=(websocket,))
    bot_thread.start()
    bot_thread.join()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
