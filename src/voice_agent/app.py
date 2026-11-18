"""FastAPI relay: serves the browser client and bridges /ws <-> Realtime API session."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from voice_agent.config import Settings
from voice_agent.handoff import HandoffQueue
from voice_agent.metrics import MetricsSink, configure_logging
from voice_agent.orchestrator import Orchestrator
from voice_agent.realtime import RealtimeConnection
from voice_agent.realtime import connect as connect_realtime

logger = logging.getLogger("voice_agent.app")

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


class WebSocketClient:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.closed = False

    async def send_json(self, data: dict[str, Any]) -> None:
        if self.closed:
            return
        try:
            await self.ws.send_text(json.dumps(data))
        except Exception:  # noqa: BLE001 - client went away
            self.closed = True


def create_app(settings: Settings | None = None, realtime_factory=None) -> FastAPI:
    settings = settings or Settings.from_env()
    factory = realtime_factory or (lambda: connect_realtime(settings))
    app = FastAPI(title="Realtime voice agent demo")
    app.state.settings = settings
    app.state.handoffs = HandoffQueue(settings.handoff_queue_path or None)
    app.state.metrics = MetricsSink()

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "provider": settings.provider}

    @app.get("/api/handoffs")
    async def handoffs() -> list[dict[str, Any]]:
        return app.state.handoffs.recent()

    @app.get("/api/metrics")
    async def metrics() -> list[dict[str, Any]]:
        return app.state.metrics.records[-200:]

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        client = WebSocketClient(ws)
        try:
            realtime: RealtimeConnection = await factory()
        except Exception as exc:  # noqa: BLE001
            logger.exception("realtime connect failed")
            await client.send_json({"type": "error", "error": {"message": str(exc)}})
            await ws.close()
            return
        orch = Orchestrator(
            realtime,
            client,
            settings,
            handoff_queue=app.state.handoffs,
            metrics=app.state.metrics,
        )
        pump = asyncio.create_task(orch.run())
        try:
            while not orch.ended:
                recv = asyncio.create_task(ws.receive())
                done, _ = await asyncio.wait({recv, pump}, return_when=asyncio.FIRST_COMPLETED)
                if recv not in done:
                    recv.cancel()
                    break
                msg = recv.result()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    await orch.handle_client_audio(msg["bytes"])
                elif msg.get("text"):
                    await orch.handle_client_message(json.loads(msg["text"]))
        except WebSocketDisconnect:
            pass
        finally:
            await orch.finish(orch.end_reason or "client_disconnected")
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump
            client.closed = True
            with contextlib.suppress(Exception):
                await ws.close()

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app


def main() -> None:
    import uvicorn

    load_dotenv()
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    settings.validate()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
