from __future__ import annotations

import asyncio
from typing import Any

import pytest

from voice_agent.audio import to_b64, tone
from voice_agent.config import Settings
from voice_agent.handoff import HandoffQueue
from voice_agent.metrics import MetricsSink
from voice_agent.orchestrator import Orchestrator
from voice_agent.tools import Tool, ToolRegistry, default_registry


class ManualClock:
    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeClient:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def send_json(self, data: dict[str, Any]) -> None:
        self.messages.append(data)

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [m for m in self.messages if m["type"] == kind]


class FakeRealtime:
    """Records everything the orchestrator sends; tests inject server events directly."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self._q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def send(self, event: dict[str, Any]) -> None:
        self.sent.append(event)

    async def recv(self) -> dict[str, Any] | None:
        return await self._q.get()

    async def close(self) -> None:
        self.closed = True
        self._q.put_nowait(None)

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.sent if e["type"] == kind]

    def types(self) -> list[str]:
        return [e["type"] for e in self.sent]


def audio_ms(ms: int) -> str:
    return to_b64(tone(ms))


def slow_registry(order_latency_s: float) -> ToolRegistry:
    async def order_status(args: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(order_latency_s)
        return {"order_id": args["order_id"], "found": True, "status": "shipped"}

    base = default_registry(0, 0)
    return ToolRegistry(
        [
            Tool("order_status", "x", {"type": "object"}, order_status),
            base.get("book_callback"),  # type: ignore[list-item]
            base.get("escalate_to_human"),  # type: ignore[list-item]
        ]
    )


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def realtime() -> FakeRealtime:
    return FakeRealtime()


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def make_orch(realtime: FakeRealtime, client: FakeClient, clock: ManualClock):
    def _make(**overrides: Any) -> Orchestrator:
        tools = overrides.pop("tools", None) or default_registry(0, 0)
        clk = overrides.pop("clock", clock)
        settings = Settings(**{"filler_threshold_ms": 700, **overrides})
        return Orchestrator(
            realtime,
            client,
            settings,
            tools=tools,
            handoff_queue=HandoffQueue(),
            metrics=MetricsSink(),
            clock=clk,
            session_id="test-session",
        )

    return _make
