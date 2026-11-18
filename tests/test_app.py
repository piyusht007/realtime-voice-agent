"""FastAPI relay wiring: health, static client and the /ws bridge in mock mode."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from voice_agent.app import create_app
from voice_agent.config import Settings
from voice_agent.realtime.mock import MockRealtimeConnection


def make_client():
    settings = Settings(provider="mock", mock_order_latency_ms=0)

    async def factory():
        return MockRealtimeConnection(speed=1000, think_ms=0)

    return TestClient(create_app(settings, realtime_factory=factory))


def test_health_and_index():
    with make_client() as c:
        assert c.get("/healthz").json() == {"ok": True, "provider": "mock"}
        assert "Realtime voice agent" in c.get("/").text
        assert c.get("/static/audio/capture-worklet.js").status_code == 200


def test_websocket_round_trip_and_handoff():
    with make_client() as c, c.websocket_connect("/ws") as ws:
        assert json.loads(ws.receive_text())["type"] in {"state", "session.started"}
        ws.send_text(json.dumps({"type": "text", "text": "where is my order 1042?"}))
        seen = []
        while True:
            msg = json.loads(ws.receive_text())
            seen.append(msg["type"])
            if msg["type"] == "transcript" and msg["role"] == "assistant" and msg["final"]:
                assert "shipped" in msg["text"]
                break
        assert "audio" in seen and "tool" in seen

        ws.send_text(json.dumps({"type": "text", "text": "get me a human"}))
        while True:
            msg = json.loads(ws.receive_text())
            if msg["type"] == "session.ended":
                assert msg["reason"] == "handoff"
                break
        handoffs = c.get("/api/handoffs").json()
        assert handoffs and handoffs[-1]["intent"] in {"order_status", "general_support"}
        assert c.get("/api/metrics").json()
