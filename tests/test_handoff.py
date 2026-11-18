"""Handoff to a human: summary to the queue, farewell, then the AI session ends."""

from __future__ import annotations

import asyncio
import json

from voice_agent.handoff import HandoffQueue, classify_intent, score_sentiment
from voice_agent.orchestrator import Phase


async def say_user(orch, text, item="u"):
    await orch.handle_realtime_event({"type": "input_audio_buffer.speech_stopped"})
    await orch.handle_realtime_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item,
            "transcript": text,
        }
    )


async def escalate(orch, reason="caller asked for a person"):
    await orch.handle_realtime_event({"type": "response.created", "response": {"id": "r_esc"}})
    await orch.handle_realtime_event(
        {
            "type": "response.function_call_arguments.done",
            "call_id": "c_esc",
            "name": "escalate_to_human",
            "arguments": json.dumps({"reason": reason}),
        }
    )


async def test_handoff_publishes_summary_and_ends_session(make_orch, realtime, client):
    orch = make_orch()
    await say_user(orch, "Where is my order 1042? It is late again and I'm frustrated.")
    await say_user(orch, "This is unacceptable, let me talk to a human.", item="u2")
    await escalate(orch)

    assert orch.phase == Phase.HANDING_OFF
    summary = await asyncio.wait_for(orch.handoff_queue.get(), 1)
    assert summary is orch.handoff_summary
    assert summary["ticket_id"].startswith("HO-")
    assert summary["intent"] == "order_status"
    assert summary["sentiment"]["label"] == "negative"
    assert [e["role"] for e in summary["transcript"]] == ["user", "user"]
    assert summary["tools_used"] == ["escalate_to_human"]
    assert client.of_type("handoff")[0]["summary"]["ticket_id"] == summary["ticket_id"]

    out = realtime.of_type("conversation.item.create")[0]["item"]
    assert out["call_id"] == "c_esc"
    assert json.loads(out["output"]) == {
        "status": "transferring",
        "ticket_id": summary["ticket_id"],
    }

    # Mic audio is no longer forwarded once a handoff is underway.
    n = len(realtime.sent)
    await orch.handle_client_audio(b"\x00\x01" * 100)
    assert len(realtime.sent) == n

    # Escalation response completes -> queued farewell is created.
    await orch.handle_realtime_event({"type": "response.done", "response": {"id": "r_esc"}})
    farewell = realtime.sent[-1]
    assert farewell["type"] == "response.create"
    assert farewell["response"]["metadata"]["purpose"] == "farewell"
    assert not orch.ended

    await orch.handle_realtime_event(
        {
            "type": "response.created",
            "response": {"id": "r_bye", "metadata": {"purpose": "farewell"}},
        }
    )
    await orch.handle_realtime_event(
        {"type": "response.done", "response": {"id": "r_bye", "metadata": {"purpose": "farewell"}}}
    )
    assert orch.ended and orch.end_reason == "handoff"
    assert realtime.closed
    assert client.of_type("session.ended")[-1]["reason"] == "handoff"


async def test_handoff_ends_even_if_farewell_never_completes(make_orch, realtime, client):
    orch = make_orch(handoff_farewell_timeout_s=0.05)
    await escalate(orch)
    await asyncio.wait_for(orch.wait_ended(), 1)
    assert orch.end_reason == "handoff" and realtime.closed


async def test_handoff_queue_mirrors_to_jsonl(tmp_path):
    path = tmp_path / "q" / "handoffs.jsonl"
    q = HandoffQueue(str(path))
    await q.publish({"ticket_id": "HO-1"})
    await q.publish({"ticket_id": "HO-2"})
    lines = path.read_text().splitlines()
    assert [json.loads(x)["ticket_id"] for x in lines] == ["HO-1", "HO-2"]
    assert [x["ticket_id"] for x in q.recent()] == ["HO-1", "HO-2"]
    assert q.qsize() == 2


def test_sentiment_and_intent_heuristics():
    assert score_sentiment(["thanks, that was great"])["label"] == "positive"
    assert score_sentiment(["this is terrible and wrong"])["label"] == "negative"
    assert score_sentiment(["what time is it"])["label"] == "neutral"
    assert classify_intent(["I was charged twice"], []) == "billing_or_refund"
    assert classify_intent(["hello"], ["book_callback"]) == "callback_request"
    assert classify_intent(["hello"], []) == "general_support"
