"""Latency-prompt pattern: speak a filler only when a tool is slower than the threshold."""

from __future__ import annotations

import asyncio
import time

from conftest import slow_registry


async def call_tool(orch, name="order_status", args='{"order_id": "1042"}', call_id="call_1"):
    await orch.handle_realtime_event({"type": "input_audio_buffer.speech_stopped"})
    await orch.handle_realtime_event({"type": "response.created", "response": {"id": "r_fc"}})
    await orch.handle_realtime_event(
        {
            "type": "response.function_call_arguments.done",
            "call_id": call_id,
            "name": name,
            "arguments": args,
        }
    )


async def drain_tools(orch):
    while orch._tool_tasks:
        await asyncio.gather(*list(orch._tool_tasks))


def filler_creates(realtime):
    return [
        e
        for e in realtime.of_type("response.create")
        if e["response"].get("metadata", {}).get("purpose") == "filler"
    ]


async def test_fast_tool_gets_no_filler(make_orch, realtime, client):
    orch = make_orch(tools=slow_registry(0.0), filler_threshold_ms=200, clock=time.monotonic)
    await call_tool(orch)
    await orch.handle_realtime_event(
        {"type": "response.done", "response": {"id": "r_fc", "status": "completed"}}
    )
    await drain_tools(orch)

    assert filler_creates(realtime) == []
    out = realtime.of_type("conversation.item.create")[0]["item"]
    assert out["type"] == "function_call_output" and out["call_id"] == "call_1"
    assert realtime.types()[-1] == "response.create"
    assert client.of_type("filler") == []


async def test_slow_tool_plays_filler_then_answer(make_orch, realtime, client):
    orch = make_orch(tools=slow_registry(0.25), filler_threshold_ms=50, clock=time.monotonic)
    await call_tool(orch)
    await orch.handle_realtime_event(
        {"type": "response.done", "response": {"id": "r_fc", "status": "completed"}}
    )
    await asyncio.sleep(0.1)  # past the threshold, tool still running

    fillers = filler_creates(realtime)
    assert len(fillers) == 1
    resp = fillers[0]["response"]
    assert resp["conversation"] == "none" and resp["tool_choice"] == "none"
    assert "One moment" in resp["instructions"]
    assert "conversation.item.create" not in realtime.types()  # answer not ready yet

    # Filler is speaking while the tool finishes -> the answer waits its turn.
    await orch.handle_realtime_event(
        {
            "type": "response.created",
            "response": {"id": "r_fill", "metadata": {"purpose": "filler"}},
        }
    )
    await drain_tools(orch)
    assert realtime.types()[-1] == "conversation.item.create"
    await orch.handle_realtime_event(
        {
            "type": "response.done",
            "response": {"id": "r_fill", "status": "completed", "metadata": {"purpose": "filler"}},
        }
    )
    assert realtime.types()[-1] == "response.create"
    assert realtime.sent[-1]["response"]["metadata"]["purpose"] == "answer"

    finished = client.of_type("tool")[-1]
    assert finished["status"] == "finished" and finished["filler"] is True
    assert finished["duration_ms"] >= 200


async def test_filler_counts_as_first_audio_in_metrics(make_orch, realtime, client, clock):
    orch = make_orch(tools=slow_registry(0.1), filler_threshold_ms=10)
    await call_tool(orch)
    await orch.handle_realtime_event({"type": "response.done", "response": {"id": "r_fc"}})
    await asyncio.sleep(0.03)
    clock.advance(0.3)
    await orch.handle_realtime_event(
        {"type": "response.created", "response": {"id": "rf", "metadata": {"purpose": "filler"}}}
    )
    await orch.handle_realtime_event(
        {"type": "response.output_item.added", "item": {"id": "f1", "type": "message"}}
    )
    await orch.handle_realtime_event({"type": "response.audio.delta", "item_id": "f1", "delta": ""})
    await drain_tools(orch)
    await orch.handle_realtime_event(
        {"type": "response.done", "response": {"id": "rf", "metadata": {"purpose": "filler"}}}
    )
    clock.advance(0.9)
    await orch.handle_realtime_event({"type": "response.created", "response": {"id": "ra"}})
    await orch.handle_realtime_event(
        {"type": "response.output_item.added", "item": {"id": "a1", "type": "message"}}
    )
    await orch.handle_realtime_event({"type": "response.audio.delta", "item_id": "a1", "delta": ""})
    await orch.handle_realtime_event({"type": "response.done", "response": {"id": "ra"}})

    rec = orch.metrics.records[-1]
    assert rec["first_audio_latency_ms"] == 300.0
    assert rec["first_audio_was_filler"] is True
    assert rec["answer_latency_ms"] == 1200.0
    assert rec["tool_calls"][0]["name"] == "order_status"
    assert rec["tool_calls"][0]["filler_played"] is True
    assert client.of_type("metrics")[-1]["turn_id"] == 1


async def test_stale_queued_filler_is_dropped(make_orch, realtime):
    # Function-call response is still streaming when the threshold passes, so the filler is
    # queued; the tool then finishes before it could play -> filler must never be sent.
    orch = make_orch(tools=slow_registry(0.08), filler_threshold_ms=20, clock=time.monotonic)
    await call_tool(orch)
    await drain_tools(orch)
    assert filler_creates(realtime) == []
    await orch.handle_realtime_event({"type": "response.done", "response": {"id": "r_fc"}})
    assert filler_creates(realtime) == []
    assert realtime.sent[-1]["response"]["metadata"]["purpose"] == "answer"
