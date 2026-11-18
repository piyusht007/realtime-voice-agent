"""Tool registry behaviour and orchestrator tool dispatch."""

from __future__ import annotations

import asyncio
import json

from voice_agent.tools import Tool, ToolRegistry, default_registry


async def test_order_status_lookup():
    reg = default_registry(0, 0)
    assert (await reg.call("order_status", '{"order_id": "#1042"}'))["status"] == "shipped"
    assert (await reg.call("order_status", {"order_id": "9999"}))["found"] is False


async def test_book_callback_returns_reference():
    res = await default_registry(0, 0).call("book_callback", {"preferred_time": "tomorrow 10am"})
    assert res["booked"] is True and res["slot"] == "tomorrow 10am"
    assert res["reference"].startswith("CB-")


async def test_errors_are_returned_not_raised():
    async def boom(_args):
        raise RuntimeError("db down")

    reg = ToolRegistry([Tool("boom", "", {}, boom)])
    assert "RuntimeError" in (await reg.call("boom", "{}"))["error"]
    assert "unknown tool" in (await reg.call("nope", "{}"))["error"]
    assert "JSON" in (await reg.call("boom", "{not json"))["error"]


def test_schemas_exposed_to_model():
    names = [s["name"] for s in default_registry().schemas()]
    assert names == ["order_status", "book_callback", "escalate_to_human"]
    assert all(s["type"] == "function" for s in default_registry().schemas())


async def test_session_config_registers_tools_and_vad(make_orch, realtime):
    orch = make_orch(vad_silence_ms=650)
    await orch.start()
    session = realtime.sent[0]["session"]
    assert realtime.sent[0]["type"] == "session.update"
    assert session["input_audio_format"] == session["output_audio_format"] == "pcm16"
    assert session["turn_detection"]["type"] == "server_vad"
    assert session["turn_detection"]["silence_duration_ms"] == 650
    assert session["turn_detection"]["interrupt_response"] is False
    assert {t["name"] for t in session["tools"]} == {
        "order_status",
        "book_callback",
        "escalate_to_human",
    }


async def test_dispatch_routes_result_back_to_model(make_orch, realtime, client):
    orch = make_orch()
    await orch.handle_realtime_event({"type": "response.created", "response": {"id": "r"}})
    await orch.handle_realtime_event(
        {
            "type": "response.function_call_arguments.done",
            "call_id": "c42",
            "name": "book_callback",
            "arguments": '{"preferred_time": "friday 3pm"}',
        }
    )
    await asyncio.gather(*list(orch._tool_tasks))
    out = realtime.of_type("conversation.item.create")[0]["item"]
    assert out == {"type": "function_call_output", "call_id": "c42", "output": out["output"]}
    assert json.loads(out["output"])["slot"] == "friday 3pm"
    assert [m["status"] for m in client.of_type("tool")] == ["started", "finished"]
    assert orch.tools_used == ["book_callback"]


async def test_response_create_is_queued_while_response_active(make_orch, realtime):
    orch = make_orch()
    await orch.handle_realtime_event({"type": "response.created", "response": {"id": "r"}})
    await orch._request_response({"type": "response.create", "response": {}})
    assert realtime.of_type("response.create") == []
    await orch.handle_realtime_event({"type": "response.done", "response": {"id": "r"}})
    assert len(realtime.of_type("response.create")) == 1


async def test_benign_errors_not_forwarded(make_orch, client):
    orch = make_orch()
    await orch.handle_realtime_event(
        {"type": "error", "error": {"code": "response_cancel_not_active"}}
    )
    await orch.handle_realtime_event({"type": "error", "error": {"code": "rate_limit"}})
    assert [m["error"]["code"] for m in client.of_type("error")] == ["rate_limit"]
