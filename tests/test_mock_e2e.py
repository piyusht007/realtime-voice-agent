"""End-to-end against the deterministic mock Realtime server (no API key needed)."""

from __future__ import annotations

import asyncio

from conftest import FakeClient
from voice_agent.audio import silence, tone
from voice_agent.config import Settings
from voice_agent.handoff import HandoffQueue
from voice_agent.metrics import MetricsSink
from voice_agent.orchestrator import Orchestrator
from voice_agent.realtime.mock import MockRealtimeConnection
from voice_agent.tools import default_registry


def make(speed=1000.0, order_ms=0, threshold_ms=700, script=None):
    mock = MockRealtimeConnection(speed=speed, think_ms=0, script=script)
    client = FakeClient()
    orch = Orchestrator(
        mock,
        client,
        Settings(filler_threshold_ms=threshold_ms),
        tools=default_registry(order_ms, 0),
        handoff_queue=HandoffQueue(),
        metrics=MetricsSink(),
    )
    return orch, mock, client


async def wait_for(pred, timeout=3.0):
    async def _poll():
        while not pred():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), timeout)


def final_assistant(client):
    return [
        m["text"]
        for m in client.of_type("transcript")
        if m["role"] == "assistant" and m["final"] and not m.get("filler")
    ]


async def test_full_conversation_with_filler_and_handoff():
    orch, mock, client = make(order_ms=120, threshold_ms=30)
    pump = asyncio.create_task(orch.run())
    await wait_for(lambda: orch.phase == "listening")

    await orch.handle_client_message({"type": "text", "text": "Where is my order 1042?"})
    await wait_for(lambda: any("shipped" in t for t in final_assistant(client)))
    assert client.of_type("filler"), "slow order lookup should trigger a latency prompt"
    filler_items = [m for m in client.of_type("transcript") if m.get("filler")]
    assert filler_items and "One moment" in filler_items[0]["text"]
    await wait_for(lambda: len(orch.metrics.records) == 1)
    rec = orch.metrics.records[0]
    assert rec["first_audio_was_filler"] is True
    assert rec["tool_calls"][0]["filler_played"] is True

    await orch.handle_client_message(
        {"type": "text", "text": "Please book a callback for tomorrow at 10am"}
    )
    await wait_for(lambda: any("callback tomorrow at 10am" in t for t in final_assistant(client)))

    await orch.handle_client_message({"type": "text", "text": "I want to talk to a human"})
    await asyncio.wait_for(pump, 3)
    assert orch.end_reason == "handoff"
    summary = orch.handoff_summary
    assert summary["tools_used"] == ["order_status", "book_callback", "escalate_to_human"]
    assert any("transferring" in t for t in final_assistant(client))
    # Out-of-band filler never pollutes the model's conversation history.
    assert not any("One moment" in (i.get("text") or "") for i in mock.items)


async def test_voice_turn_via_mock_vad():
    orch, mock, client = make(script=["Where is my order 3141?"])
    pump = asyncio.create_task(orch.run())
    await wait_for(lambda: orch.phase == "listening")
    for chunk in [tone(400, amplitude=0.3), silence(600)]:
        for i in range(0, len(chunk), 1920):  # 40 ms frames like the browser worklet
            await orch.handle_client_audio(chunk[i : i + 1920])
    await wait_for(lambda: any("delivered" in t for t in final_assistant(client)))
    user = [m for m in client.of_type("transcript") if m["role"] == "user"]
    assert user[0]["text"] == "Where is my order 3141?"
    await wait_for(lambda: orch.metrics.records)
    assert orch.metrics.records[0]["source"] == "voice"
    await orch.finish("test")
    await asyncio.wait_for(pump, 1)


async def test_barge_in_against_mock_truncates_conversation_item():
    orch, mock, client = make(speed=5.0)  # slow enough to interrupt mid-stream
    pump = asyncio.create_task(orch.run())
    await wait_for(lambda: orch.phase == "listening")
    await orch.handle_client_message({"type": "text", "text": "hello there"})
    await wait_for(lambda: len(client.of_type("audio")) >= 2)

    loud = tone(100, amplitude=0.4)
    await orch.handle_client_audio(loud)  # mock VAD -> speech_started
    await wait_for(lambda: client.of_type("audio.flush"))

    assert any(e["type"] == "response.cancel" for e in mock.sent)
    truncated = [i for i in mock.items if "truncated_at_ms" in i]
    assert len(truncated) == 1 and truncated[0]["role"] == "assistant"
    assert truncated[0]["truncated_at_ms"] == client.of_type("audio.flush")[0]["played_ms"]
    await wait_for(lambda: orch.metrics.records)
    assert orch.metrics.records[0]["barge_in"] is True
    await orch.finish("test")
    await asyncio.wait_for(pump, 1)
