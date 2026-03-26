"""Application state — sessions, media, broadcast channels."""

from __future__ import annotations

import asyncio
from typing import Any

from .models import SessionInfo


class SessionContext:
    """Runtime context for a running session."""

    def __init__(self, info: SessionInfo):
        self.info = info
        # Broadcast queue for subtitle messages to WebSocket clients
        self.subscribers: list[asyncio.Queue] = []
        # Task handle for the transcription runner
        self.task: asyncio.Task | None = None
        # Cancellation event
        self.cancel_event = asyncio.Event()
        # Audio track for WebRTC (created by session runner, used by WS handler)
        self.audio_track: Any = None
        # Event signaling that a WebRTC client is ready
        self.client_ready = asyncio.Event()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self.subscribers.remove(q)
        except ValueError:
            pass

    async def broadcast(self, msg: dict[str, Any]) -> None:
        for q in self.subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                # Drop oldest message on overflow
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    pass


class AppState:
    """Global application state singleton."""

    def __init__(self) -> None:
        self.sessions: dict[str, SessionContext] = {}

    def add_session(self, info: SessionInfo) -> SessionContext:
        ctx = SessionContext(info)
        self.sessions[info.id] = ctx
        return ctx

    def get_session(self, session_id: str) -> SessionContext | None:
        return self.sessions.get(session_id)

    def remove_session(self, session_id: str) -> None:
        ctx = self.sessions.pop(session_id, None)
        if ctx and ctx.task and not ctx.task.done():
            ctx.cancel_event.set()
            ctx.task.cancel()

    def list_sessions(self) -> list[SessionInfo]:
        return [ctx.info for ctx in self.sessions.values()]


app_state = AppState()
