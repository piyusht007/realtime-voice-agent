"""Barge-in: cancel, truncate to what was actually heard, flush the client."""

from __future__ import annotations

import pytest

from conftest import audio_ms
from voice_agent.orchestrator import Phase


async def start_assistant_audio(orch, clock, *, item_id="item_a", purpose=None, ms=(500, 500)):
    await orch.handle_realtime_event({"type": "input_audio_buffer.speech_stopped"})
    clock.advance(0.2)
    meta = {"metadata": {"purpose": purpose}} if purpose else {}
    await orch.handle_realtime_event({"type": "response.created", "response": {"id": "r1", **meta}})
    await orch.handle_realtime_event(
        {"type": "response.output_item.added", "item": {"id": item_id, "type": "message"}}
    )
    for chunk in ms:
        await orch.handle_realtime_event(
            {"type": "response.audio.delta", "item_id": item_id, "delta": audio_ms(chunk)}
        )
    await orch.handle_realtime_event(
        {
            "type": "response.audio_transcript.delta",
            "item_id": item_id,
            "delta": "Your order has shipped and",
        }
    )


async def test_barge_in_cancels_truncates_and_flushes(make_orch, realtime, client, clock):
    orch = make_orch()
    await start_assistant_audio(orch, clock)
    assert orch.phase == Phase.ASSISTANT_SPEAKING

    clock.advance(0.4)  # caller has heard ~400 ms of the 1000 ms generated
    await orch.handle_realtime_event({"type": "input_audio_buffer.speech_started"})

    assert realtime.types()[-2:] == ["response.cancel", "conversation.item.truncate"]
    trunc = realtime.of_type("conversation.item.truncate")[0]
    assert trunc == {
        "type": "conversation.item.truncate",
        "item_id": "item_a",
        "content_index": 0,
        "audio_end_ms": 400,
    }
    flush = client.of_type("audio.flush")[0]
    assert flush["item_id"] == "item_a" and flush["played_ms"] == 400
    assert orch.phase == Phase.USER_SPEAKING

    record = orch.metrics.records[-1]
    assert record["barge_in"] is True and record["outcome"] == "interrupted"
    assert record["first_audio_latency_ms"] == 200.0
    assert orch.transcript[-1].text.endswith("[interrupted]")


async def test_client_reported_progress_is_preferred(make_orch, realtime, client, clock):
    orch = make_orch()
    await start_assistant_audio(orch, clock)
    clock.advance(0.5)
    # Playback started late on the client (jitter buffer): only 250 ms heard so far.
    await orch.handle_client_message(
        {"type": "playback.progress", "item_id": "item_a", "played_ms": 250}
    )
    clock.advance(0.05)
    await orch.handle_realtime_event({"type": "input_audio_buffer.speech_started"})
    assert realtime.of_type("conversation.item.truncate")[0]["audio_end_ms"] == 300


async def test_no_truncate_when_audio_already_finished(make_orch, realtime, client, clock):
    orch = make_orch()
    await start_assistant_audio(orch, clock)
    await orch.handle_realtime_event(
        {"type": "response.done", "response": {"id": "r1", "status": "completed"}}
    )
    clock.advance(2.0)  # all 1000 ms have played out
    await orch.handle_realtime_event({"type": "input_audio_buffer.speech_started"})
    assert "response.cancel" not in realtime.types()
    assert "conversation.item.truncate" not in realtime.types()
    assert client.of_type("audio.flush") == []


async def test_played_ms_capped_at_generated_audio(make_orch, realtime, clock):
    orch = make_orch()
    await start_assistant_audio(orch, clock, ms=(300,))
    clock.advance(0.1)
    track = orch._tracks["item_a"]
    assert orch.played_ms(track) == pytest.approx(100)
    clock.advance(5)
    assert orch.played_ms(track) == 300


async def test_stale_audio_after_barge_in_is_dropped(make_orch, realtime, client, clock):
    orch = make_orch()
    await start_assistant_audio(orch, clock)
    clock.advance(0.1)
    await orch.handle_realtime_event({"type": "input_audio_buffer.speech_started"})
    before = len(client.of_type("audio"))
    await orch.handle_realtime_event(
        {"type": "response.audio.delta", "item_id": "item_a", "delta": audio_ms(100)}
    )
    assert len(client.of_type("audio")) == before


async def test_filler_barge_in_flushes_without_truncating(make_orch, realtime, client, clock):
    orch = make_orch()
    await start_assistant_audio(orch, clock, item_id="filler_1", purpose="filler")
    clock.advance(0.1)
    await orch.handle_realtime_event({"type": "input_audio_buffer.speech_started"})
    assert "response.cancel" in realtime.types()
    # Out-of-band filler items are not in the conversation, so nothing to truncate.
    assert "conversation.item.truncate" not in realtime.types()
    assert client.of_type("audio.flush")[0]["item_id"] == "filler_1"


async def test_text_input_also_interrupts(make_orch, realtime, client, clock):
    orch = make_orch()
    await start_assistant_audio(orch, clock)
    clock.advance(0.3)
    await orch.handle_client_message({"type": "text", "text": "stop, different question"})
    types = realtime.types()
    assert types.index("conversation.item.truncate") < types.index("conversation.item.create")
    # The new response is queued until the cancelled one reports response.done.
    assert types[-1] == "conversation.item.create"
    await orch.handle_realtime_event(
        {"type": "response.done", "response": {"id": "r1", "status": "cancelled"}}
    )
    assert realtime.types()[-1] == "response.create"
