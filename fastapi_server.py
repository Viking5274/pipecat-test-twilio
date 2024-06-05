from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from starlette.responses import HTMLResponse

app = FastAPI()


@app.post('/start_call')
async def start_call():
    print("POST TwiML")
    return HTMLResponse(content=open("templates/streams.xml").read(), media_type="application/xml")

