"""Voice-agent orchestration state machine.

Sits between the browser client and a Realtime API session and owns the parts of the
conversation the model cannot do on its own:

* **barge-in** - on ``speech_started`` cancel the in-flight response, truncate the assistant
  item to what the caller actually heard, and tell the client to flush its playback queue;
* **latency prompts** - if a tool call takes longer than ``filler_threshold_ms`` speak a short
  out-of-band filler ("one moment...") so the line never goes dead, then the real answer;
* **tool dispatch** - run mock async tools and feed results back as ``function_call_output``;
* **handoff** - ``escalate_to_human`` publishes a summary (transcript, intent, sentiment) to a
  queue, lets the model say goodbye, then ends the AI session;
* **metrics** - per-turn speech-end -> first-audio-byte latency as structured JSON.

All I/O goes through two narrow interfaces (``RealtimeConnection`` and ``ClientSink``) so the
whole thing is unit-testable with fakes and a deterministic clock.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from voice_agent.audio import b64_duration_ms
from voice_agent.config import Settings
from voice_agent.handoff import HandoffQueue, TranscriptEntry, build_handoff_summary
from voice_agent.metrics import MetricsSink, ToolMetric, TurnMetrics
from voice_agent.prompts import SYSTEM_INSTRUCTIONS, filler_instructions
from voice_agent.realtime.base import (
    AUDIO_DELTA_EVENTS,
    AUDIO_TRANSCRIPT_DELTA_EVENTS,
    AUDIO_TRANSCRIPT_DONE_EVENTS,
    RealtimeConnection,
)
from voice_agent.tools import ESCALATE_TOOL, ToolRegistry, default_registry

logger = logging.getLogger(__name__)

BENIGN_ERROR_CODES = {"response_cancel_not_active"}


class ClientSink(Protocol):
    async def send_json(self, data: dict[str, Any]) -> None: ...


class Phase(StrEnum):
    CONNECTING = "connecting"
    LISTENING = "listening"
    USER_SPEAKING = "user_speaking"
    THINKING = "thinking"
    TOOL_RUNNING = "tool_running"
    ASSISTANT_SPEAKING = "assistant_speaking"
    HANDING_OFF = "handing_off"
    ENDED = "ended"


@dataclass
class AudioTrack:
    """Bookkeeping for one assistant audio item streamed to the client."""

    item_id: str
    purpose: str
    first_delta_at: float | None = None
    sent_ms: float = 0.0
    reported_played_ms: float | None = None
    reported_at: float | None = None
    text: str = ""
    truncated: bool = False
    recorded: bool = False

    @property
    def is_filler(self) -> bool:
        return self.purpose == "filler"


def _purpose(event: dict[str, Any]) -> str:
    return ((event.get("response") or {}).get("metadata") or {}).get("purpose", "answer")


class Orchestrator:
    def __init__(
        self,
        realtime: RealtimeConnection,
        client: ClientSink,
        settings: Settings | None = None,
        *,
        tools: ToolRegistry | None = None,
        handoff_queue: HandoffQueue | None = None,
        metrics: MetricsSink | None = None,
        clock: Callable[[], float] = time.monotonic,
        session_id: str | None = None,
    ):
        self.settings = settings or Settings()
        self.realtime = realtime
        self.client = client
        self.tools = tools or default_registry(
            self.settings.mock_order_latency_ms, self.settings.mock_callback_latency_ms
        )
        self.handoff_queue = handoff_queue or HandoffQueue()
        self.metrics = metrics or MetricsSink()
        self.clock = clock
        self.session_id = session_id or uuid.uuid4().hex[:12]

        self.phase = Phase.CONNECTING
        self.transcript: list[TranscriptEntry] = []
        self.tools_used: list[str] = []
        self.handoff_summary: dict[str, Any] | None = None
        self.end_reason: str | None = None

        self._response_active = False
        self._active_purpose: str | None = None
        self._pending: deque[dict[str, Any]] = deque()
        self._audio: AudioTrack | None = None
        self._tracks: dict[str, AudioTrack] = {}
        self._tool_tasks: set[asyncio.Task[None]] = set()
        self._turn: TurnMetrics | None = None
        self._turn_counter = 0
        self._latencies: list[float] = []
        self._handoff_pending = False
        self._farewell_timer: asyncio.Task[None] | None = None
        self._ended = asyncio.Event()

    # ================================================================ lifecycle
    def session_config(self) -> dict[str, Any]:
        s = self.settings
        return {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": SYSTEM_INSTRUCTIONS,
                "voice": s.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": s.transcription_model},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": s.vad_threshold,
                    "prefix_padding_ms": s.vad_prefix_padding_ms,
                    "silence_duration_ms": s.vad_silence_ms,
                    "create_response": True,
                    # Barge-in is handled explicitly below so truncation is exact.
                    "interrupt_response": False,
                },
                "tools": self.tools.schemas(),
                "tool_choice": "auto",
            },
        }

    async def start(self) -> None:
        await self.realtime.send(self.session_config())
        await self._set_phase(Phase.LISTENING)
        await self.client.send_json({"type": "session.started", "session_id": self.session_id})

    async def run(self) -> None:
        """Start the session and pump realtime events until the session ends."""
        await self.start()
        while not self._ended.is_set():
            event = await self.realtime.recv()
            if event is None:
                break
            await self.handle_realtime_event(event)
        await self.finish("realtime_closed")

    @property
    def ended(self) -> bool:
        return self._ended.is_set()

    async def wait_ended(self) -> None:
        await self._ended.wait()

    async def finish(self, reason: str) -> None:
        if self._ended.is_set():
            return
        self._ended.set()
        self.end_reason = reason
        await self._close_turn(outcome="session_ended" if self._turn else "completed")
        current = asyncio.current_task()
        for task in [*self._tool_tasks, self._farewell_timer]:
            if task and task is not current and not task.done():
                task.cancel()
        self.phase = Phase.ENDED
        with contextlib.suppress(Exception):
            await self.realtime.close()
        with contextlib.suppress(Exception):
            await self.client.send_json({"type": "state", "phase": Phase.ENDED.value})
            await self.client.send_json({"type": "session.ended", "reason": reason})
        logger.info(
            json.dumps({"event": "session.ended", "session_id": self.session_id, "reason": reason})
        )

    # ================================================================ client -> agent
    async def handle_client_audio(self, pcm: bytes) -> None:
        if self.ended or self._handoff_pending:
            return
        await self.realtime.send(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}
        )

    async def handle_client_message(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "playback.progress":
            track = self._tracks.get(msg.get("item_id", ""))
            if track:
                track.reported_played_ms = float(msg.get("played_ms", 0))
                track.reported_at = self.clock()
        elif kind == "text":
            await self._on_user_text(str(msg.get("text", "")).strip())
        elif kind == "hangup":
            await self.finish("client_hangup")
        else:
            logger.debug("ignoring client message %s", kind)

    async def _on_user_text(self, text: str) -> None:
        if not text or self.ended or self._handoff_pending:
            return
        await self._interrupt()
        await self._open_turn("text")
        self._add_transcript("user", text)
        await self.realtime.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
        await self._set_phase(Phase.THINKING)
        await self._request_response(
            {"type": "response.create", "response": {"metadata": {"purpose": "answer"}}}
        )

    # ================================================================ realtime -> agent
    async def handle_realtime_event(self, event: dict[str, Any]) -> None:
        etype = event.get("type", "")
        if etype == "input_audio_buffer.speech_started":
            await self._interrupt()
            await self._set_phase(Phase.USER_SPEAKING)
        elif etype == "input_audio_buffer.speech_stopped":
            await self._open_turn("voice")
            await self._set_phase(Phase.THINKING)
        elif etype == "conversation.item.input_audio_transcription.completed":
            text = (event.get("transcript") or "").strip()
            if text:
                self._add_transcript("user", text)
                await self.client.send_json(
                    {
                        "type": "transcript",
                        "role": "user",
                        "text": text,
                        "final": True,
                        "item_id": event.get("item_id"),
                    }
                )
        elif etype == "response.created":
            self._response_active = True
            self._active_purpose = _purpose(event)
        elif etype == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "message":
                self._new_track(item["id"])
        elif etype in AUDIO_DELTA_EVENTS:
            await self._on_audio_delta(event)
        elif etype in AUDIO_TRANSCRIPT_DELTA_EVENTS:
            track = self._tracks.get(event.get("item_id", ""))
            if track and not track.truncated:
                track.text += event.get("delta", "")
                if not track.is_filler:
                    await self.client.send_json(
                        {
                            "type": "transcript",
                            "role": "assistant",
                            "text": track.text,
                            "final": False,
                            "item_id": track.item_id,
                        }
                    )
        elif etype in AUDIO_TRANSCRIPT_DONE_EVENTS:
            track = self._tracks.get(event.get("item_id", ""))
            if track and not track.truncated:
                track.text = event.get("transcript", track.text)
                self._record_assistant(track)
                await self.client.send_json(
                    {
                        "type": "transcript",
                        "role": "assistant",
                        "text": track.text,
                        "final": True,
                        "item_id": track.item_id,
                        "filler": track.is_filler,
                    }
                )
        elif etype == "response.function_call_arguments.done":
            await self._on_function_call(
                event.get("call_id", ""), event.get("name", ""), event.get("arguments", "{}")
            )
        elif etype == "response.done":
            await self._on_response_done(event)
        elif etype == "error":
            err = event.get("error") or {}
            if err.get("code") in BENIGN_ERROR_CODES:
                logger.debug("benign realtime error: %s", err)
            else:
                logger.warning(json.dumps({"event": "realtime.error", "error": err}))
                await self.client.send_json({"type": "error", "error": err})

    # ================================================================ responses
    async def _request_response(self, event: dict[str, Any]) -> None:
        """Create a response now, or queue it until the active one is done."""
        if self._response_active:
            self._pending.append(event)
            return
        self._response_active = True
        self._active_purpose = _purpose(event)
        await self.realtime.send(event)

    def _drop_pending(self, purpose: str) -> None:
        self._pending = deque(e for e in self._pending if _purpose(e) != purpose)

    async def _on_response_done(self, event: dict[str, Any]) -> None:
        has_metadata = bool((event.get("response") or {}).get("metadata"))
        finished_purpose = _purpose(event) if has_metadata else self._active_purpose
        self._response_active = False
        self._active_purpose = None
        if self._pending:
            nxt = self._pending.popleft()
            self._response_active = True
            self._active_purpose = _purpose(nxt)
            await self.realtime.send(nxt)
            return
        if self._handoff_pending and finished_purpose == "farewell":
            await self.finish("handoff")
            return
        if not self._tool_tasks and not self._handoff_pending:
            status = (event.get("response") or {}).get("status", "completed")
            await self._close_turn(outcome=status)
            await self._set_phase(Phase.LISTENING)

    # ================================================================ audio + barge-in
    def _new_track(self, item_id: str) -> AudioTrack:
        track = AudioTrack(item_id=item_id, purpose=self._active_purpose or "answer")
        self._tracks[item_id] = track
        self._audio = track
        return track

    async def _on_audio_delta(self, event: dict[str, Any]) -> None:
        item_id = event.get("item_id", "")
        track = self._tracks.get(item_id) or self._new_track(item_id)
        if track.truncated:
            return  # stale audio that raced the cancel - never forward it
        delta = event.get("delta", "")
        now = self.clock()
        if track.first_delta_at is None:
            track.first_delta_at = now
            turn = self._turn
            if turn is not None:
                if turn.first_audio_at is None:
                    turn.first_audio_at = now
                    turn.first_audio_was_filler = track.is_filler
                if not track.is_filler and turn.first_answer_audio_at is None:
                    turn.first_answer_audio_at = now
        track.sent_ms += b64_duration_ms(delta)
        await self._set_phase(Phase.ASSISTANT_SPEAKING)
        await self.client.send_json(
            {"type": "audio", "item_id": item_id, "delta": delta, "filler": track.is_filler}
        )

    def played_ms(self, track: AudioTrack) -> float:
        """Best estimate of how much of ``track`` the caller has actually heard."""
        now = self.clock()
        if track.reported_played_ms is not None and track.reported_at is not None:
            est = track.reported_played_ms + (now - track.reported_at) * 1000
        elif track.first_delta_at is not None:
            est = (now - track.first_delta_at) * 1000
        else:
            est = 0.0
        return max(0.0, min(est, track.sent_ms))

    async def _interrupt(self) -> None:
        """Barge-in: stop generation, truncate what wasn't heard, flush client playback."""
        interrupted = False
        if self._response_active:
            await self.realtime.send({"type": "response.cancel"})
            interrupted = True
        track = self._audio
        if track and not track.truncated and track.first_delta_at is not None:
            played = self.played_ms(track)
            if played < track.sent_ms - 1:
                interrupted = True
                track.truncated = True
                if not track.is_filler:
                    await self.realtime.send(
                        {
                            "type": "conversation.item.truncate",
                            "item_id": track.item_id,
                            "content_index": 0,
                            "audio_end_ms": round(played),
                        }
                    )
                    track.text = (track.text.strip() + " [interrupted]").strip()
                    self._record_assistant(track)
                await self.client.send_json(
                    {"type": "audio.flush", "item_id": track.item_id, "played_ms": round(played)}
                )
                logger.info(
                    json.dumps(
                        {
                            "event": "barge_in",
                            "session_id": self.session_id,
                            "item_id": track.item_id,
                            "played_ms": round(played),
                            "generated_ms": int(track.sent_ms),
                        }
                    )
                )
        self._drop_pending("filler")
        if self._turn is not None and interrupted:
            self._turn.barge_in = True
            if not self._tool_tasks:
                await self._close_turn(outcome="interrupted")

    # ================================================================ tools + latency prompt
    async def _on_function_call(self, call_id: str, name: str, arguments: str) -> None:
        self.tools_used.append(name)
        await self.client.send_json(
            {"type": "tool", "name": name, "status": "started", "call_id": call_id}
        )
        if name == ESCALATE_TOOL:
            await self._begin_handoff(call_id, arguments)
            return
        await self._set_phase(Phase.TOOL_RUNNING)
        task = asyncio.create_task(self._run_tool(call_id, name, arguments, self._turn))
        self._tool_tasks.add(task)
        task.add_done_callback(self._tool_tasks.discard)

    async def _run_tool(
        self, call_id: str, name: str, arguments: str, turn: TurnMetrics | None
    ) -> None:
        start = self.clock()
        call = asyncio.create_task(self.tools.call(name, arguments))
        threshold_s = self.settings.filler_threshold_ms / 1000
        done, _ = await asyncio.wait({call}, timeout=threshold_s)
        filler = False
        if not done:
            filler = True
            await self._send_filler()
        result = await call
        duration_ms = round((self.clock() - start) * 1000, 1)
        if turn is not None:
            turn.tool_calls.append(ToolMetric(name, duration_ms, filler))
        self._drop_pending("filler")  # a filler that never got to play is now stale
        if self.ended:
            return
        await self.realtime.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                },
            }
        )
        await self._request_response(
            {"type": "response.create", "response": {"metadata": {"purpose": "answer"}}}
        )
        await self.client.send_json(
            {
                "type": "tool",
                "name": name,
                "status": "finished",
                "call_id": call_id,
                "duration_ms": duration_ms,
                "filler": filler,
                "result": result,
            }
        )

    async def _send_filler(self) -> None:
        await self.client.send_json({"type": "filler", "text": self.settings.filler_text})
        await self._request_response(
            {
                "type": "response.create",
                "response": {
                    # Out-of-band: not added to the conversation, no prior context needed.
                    "conversation": "none",
                    "input": [],
                    "metadata": {"purpose": "filler"},
                    "instructions": filler_instructions(self.settings.filler_text),
                    "tool_choice": "none",
                },
            }
        )

    # ================================================================ handoff
    async def _begin_handoff(self, call_id: str, arguments: str) -> None:
        self._handoff_pending = True
        await self._set_phase(Phase.HANDING_OFF)
        try:
            reason = str(json.loads(arguments or "{}").get("reason", ""))
        except json.JSONDecodeError:
            reason = ""
        summary = build_handoff_summary(
            self.session_id, self.transcript, list(self.tools_used), reason, self._latencies
        )
        self.handoff_summary = summary
        await self.handoff_queue.publish(summary)
        logger.info(
            json.dumps(
                {
                    "event": "handoff.queued",
                    "session_id": self.session_id,
                    "ticket_id": summary["ticket_id"],
                    "intent": summary["intent"],
                    "sentiment": summary["sentiment"]["label"],
                }
            )
        )
        await self.client.send_json({"type": "handoff", "summary": summary})
        await self.realtime.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(
                        {"status": "transferring", "ticket_id": summary["ticket_id"]}
                    ),
                },
            }
        )
        await self._request_response(
            {
                "type": "response.create",
                "response": {
                    "metadata": {"purpose": "farewell"},
                    "instructions": "In one short sentence tell the caller you are transferring "
                    "them to a human colleague who will have the full context.",
                    "tool_choice": "none",
                },
            }
        )
        self._farewell_timer = asyncio.create_task(self._farewell_deadline())

    async def _farewell_deadline(self) -> None:
        await asyncio.sleep(self.settings.handoff_farewell_timeout_s)
        await self.finish("handoff")

    # ================================================================ turns + transcript
    async def _open_turn(self, source: str) -> None:
        await self._close_turn(outcome="superseded")
        self._turn_counter += 1
        self._turn = TurnMetrics(
            session_id=self.session_id,
            turn_id=self._turn_counter,
            source=source,
            speech_stopped_at=self.clock(),
        )

    async def _close_turn(self, outcome: str) -> None:
        turn = self._turn
        if turn is None:
            return
        self._turn = None
        if turn.outcome == "completed":
            turn.outcome = outcome
        if turn.first_audio_latency_ms is not None:
            self._latencies.append(turn.first_audio_latency_ms)
        record = turn.to_record()
        self.metrics.emit(record)
        with contextlib.suppress(Exception):
            await self.client.send_json({"type": "metrics", **record})

    def _add_transcript(self, role: str, text: str) -> None:
        self.transcript.append(TranscriptEntry(role=role, text=text))

    def _record_assistant(self, track: AudioTrack) -> None:
        if track.recorded or track.is_filler or not track.text.strip():
            return
        track.recorded = True
        self._add_transcript("assistant", track.text.strip())

    async def _set_phase(self, phase: Phase) -> None:
        if self.phase == phase or self.ended:
            return
        self.phase = phase
        await self.client.send_json({"type": "state", "phase": phase.value})
