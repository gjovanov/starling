"""WebSocket event fan-out for /api/ocr/ws.

Every stage transition and per-page progress tick is broadcast to all
connected clients — the frontend never polls. One JSON object per message:

  {"type": "run",        "id": 12, "document_id": 3, "status": "running",
   "engine": "glm-ocr",  "page": {"done": 2, "total": 5}}
  {"type": "extraction", "id": 7,  "run_id": 12, "status": "done",
   "validation_status": "needs_review"}
  {"type": "engine",     "name": "glm-ocr", "status": "loading|ready|unloaded"}
"""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect


class Broadcaster:
    """Per-subscriber bounded queues; drop-oldest on overflow (mirrors the
    SessionContext.broadcast pattern in vllm-server)."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    async def broadcast(self, msg: dict) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    pass


router = APIRouter()


@router.websocket("/api/ocr/ws")
async def ocr_ws(ws: WebSocket) -> None:
    from .state import app_state  # lazy: avoid import cycle at module load

    await ws.accept()
    q = app_state.ws.subscribe()

    async def forward() -> None:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=10.0)
                await ws.send_json(msg)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping"})
            except Exception:
                return

    async def receive() -> None:
        # Clients send nothing today; this loop exists to notice disconnects.
        while True:
            try:
                await ws.receive_text()
            except (WebSocketDisconnect, Exception):
                return

    t_fwd = asyncio.create_task(forward())
    t_rcv = asyncio.create_task(receive())
    try:
        _, pending = await asyncio.wait({t_fwd, t_rcv}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
    finally:
        app_state.ws.unsubscribe(q)
