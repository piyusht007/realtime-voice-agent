"""Deterministic in-process stand-in for a Realtime API session.

It speaks the same JSON event protocol as the real service closely enough to exercise the
orchestrator end-to-end without an API key:

* energy-based server VAD over appended pcm16 audio (speech_started / speech_stopped),
* scripted "transcriptions" for voice turns, keyword routing for text turns,
* streamed synthetic audio (a soft tone) + transcript deltas,
* function calls for ``order_status`` / ``book_callback`` / ``escalate_to_human``,
* ``response.cancel``, ``conversation.item.truncate`` and out-of-band (``conversation: none``)
  responses used for latency fillers,
* strictness: creating a response while one is active yields the same error the real API
  returns, which keeps the orchestrator's response queue honest.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import re
import uuid
from typing import Any

from voice_agent.audio import from_b64, pcm16_duration_ms, rms, to_b64, tone

DEFAULT_SCRIPT = [
    "Hi, where is my order 1042?",
    "Great. Can you book a callback for tomorrow morning?",
    "Actually, this is taking too long. I'd like to speak to a human please.",
]

HUMAN_WORDS = ("human", "person", "agent", "representative", "manager", "someone real")


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class MockRealtimeConnection:
    def __init__(
        self,
        *,
        speed: float = 1.0,
        think_ms: int = 250,
        chunk_ms: int = 100,
        stream_ratio: float = 0.5,
        ms_per_word: int = 280,
        vad_threshold: float = 0.02,
        silence_ms: int = 500,
        script: list[str] | None = None,
    ):
        self.speed = speed
        self.think_ms = think_ms
        self.chunk_ms = chunk_ms
        self.stream_ratio = stream_ratio  # <1 => generate faster than realtime, like the real API
        self.ms_per_word = ms_per_word
        self.vad_threshold = vad_threshold
        self.silence_ms = silence_ms
        self._script = itertools.cycle(script or DEFAULT_SCRIPT)

        self.sent: list[dict[str, Any]] = []  # every client event received (for tests)
        self.items: list[dict[str, Any]] = []
        self.session: dict[str, Any] = {}
        self._events: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._closed = False
        self._response_task: asyncio.Task[None] | None = None

        self._audio_ms = 0.0
        self._speaking = False
        self._silence_acc = 0.0
        self._speech_item: str | None = None

        self._emit({"type": "session.created", "session": {"id": _id("sess")}})

    # ------------------------------------------------------------------ transport
    async def send(self, event: dict[str, Any]) -> None:
        if self._closed:
            return
        self.sent.append(event)
        handler = getattr(self, "_on_" + event.get("type", "").replace(".", "_"), None)
        if handler is None:
            self._error("unknown_event", f"unsupported event {event.get('type')!r}")
            return
        await handler(event)

    async def recv(self) -> dict[str, Any] | None:
        if self._closed and self._events.empty():
            return None
        return await self._events.get()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._response_task
        self._events.put_nowait(None)

    # ------------------------------------------------------------------ helpers
    def _emit(self, event: dict[str, Any]) -> None:
        if not self._closed:
            event.setdefault("event_id", _id("evt"))
            self._events.put_nowait(event)

    def _error(self, code: str, message: str) -> None:
        self._emit(
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "code": code, "message": message},
            }
        )

    async def _sleep(self, ms: float) -> None:
        await asyncio.sleep(ms / 1000 / self.speed if self.speed > 0 else 0)

    @property
    def response_active(self) -> bool:
        return self._response_task is not None and not self._response_task.done()

    # ------------------------------------------------------------------ client events
    async def _on_session_update(self, event: dict[str, Any]) -> None:
        self.session.update(event.get("session", {}))
        self._emit({"type": "session.updated", "session": self.session})

    async def _on_input_audio_buffer_append(self, event: dict[str, Any]) -> None:
        pcm = from_b64(event["audio"])
        dur = pcm16_duration_ms(len(pcm))
        self._audio_ms += dur
        if rms(pcm) >= self.vad_threshold:
            self._silence_acc = 0.0
            if not self._speaking:
                self._speaking = True
                self._speech_item = _id("item")
                self._emit(
                    {
                        "type": "input_audio_buffer.speech_started",
                        "audio_start_ms": int(self._audio_ms - dur),
                        "item_id": self._speech_item,
                    }
                )
            return
        if not self._speaking:
            return
        self._silence_acc += dur
        if self._silence_acc >= self.silence_ms:
            self._speaking = False
            item_id = self._speech_item or _id("item")
            self._emit(
                {
                    "type": "input_audio_buffer.speech_stopped",
                    "audio_end_ms": int(self._audio_ms - self._silence_acc),
                    "item_id": item_id,
                }
            )
            self._emit({"type": "input_audio_buffer.committed", "item_id": item_id})
            text = next(self._script)
            item = {"id": item_id, "type": "message", "role": "user", "text": text}
            self.items.append(item)
            self._emit(
                {
                    "type": "conversation.item.created",
                    "item": {"id": item_id, "type": "message", "role": "user"},
                }
            )
            self._emit(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": item_id,
                    "content_index": 0,
                    "transcript": text,
                }
            )
            td = self.session.get("turn_detection") or {}
            if td.get("create_response", True) and not self.response_active:
                self._start_response({})

    async def _on_input_audio_buffer_clear(self, event: dict[str, Any]) -> None:
        self._emit({"type": "input_audio_buffer.cleared"})

    async def _on_conversation_item_create(self, event: dict[str, Any]) -> None:
        item = dict(event.get("item", {}))
        item.setdefault("id", _id("item"))
        if item.get("type") == "message":
            content = item.get("content") or [{}]
            item["text"] = content[0].get("text", "")
        self.items.append(item)
        self._emit({"type": "conversation.item.created", "item": item})

    async def _on_conversation_item_truncate(self, event: dict[str, Any]) -> None:
        for item in self.items:
            if item.get("id") == event.get("item_id"):
                item["truncated_at_ms"] = event.get("audio_end_ms")
                self._emit(
                    {
                        "type": "conversation.item.truncated",
                        "item_id": item["id"],
                        "content_index": event.get("content_index", 0),
                        "audio_end_ms": event.get("audio_end_ms"),
                    }
                )
                return
        self._error("item_not_found", f"no item {event.get('item_id')!r} to truncate")

    async def _on_response_create(self, event: dict[str, Any]) -> None:
        if self.response_active:
            self._error(
                "conversation_already_has_active_response",
                "Conversation already has an active response",
            )
            return
        self._start_response(event.get("response") or {})

    async def _on_response_cancel(self, event: dict[str, Any]) -> None:
        if not self.response_active:
            self._error("response_cancel_not_active", "No active response to cancel")
            return
        assert self._response_task is not None
        self._response_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._response_task

    # ------------------------------------------------------------------ responses
    def _start_response(self, params: dict[str, Any]) -> None:
        self._response_task = asyncio.create_task(self._run_response(_id("resp"), params))

    def _plan(self, params: dict[str, Any]) -> dict[str, Any]:
        if params.get("conversation") == "none":
            m = re.search(r'"([^"]+)"', params.get("instructions", ""))
            return {"kind": "say", "text": m.group(1) if m else "One moment.", "store": False}
        if (params.get("metadata") or {}).get("purpose") == "farewell":
            return {
                "kind": "say",
                "text": "Of course. I'm transferring you to a colleague now. "
                "They'll have the full context. Goodbye!",
                "store": True,
            }
        last = self.items[-1] if self.items else None
        if last and last.get("type") == "function_call_output":
            return {"kind": "say", "text": self._speak_result(last), "store": True}
        if last and last.get("type") == "message" and last.get("role") == "user":
            return self._route(last.get("text", ""))
        return {
            "kind": "say",
            "store": True,
            "text": "Hi! I can check an order, book a callback, or connect you to a person.",
        }

    def _route(self, text: str) -> dict[str, Any]:
        lower = text.lower()
        if any(w in lower for w in HUMAN_WORDS):
            return {"kind": "call", "name": "escalate_to_human", "args": {"reason": text}}
        if "order" in lower:
            m = re.search(r"\d{3,}", text)
            if m:
                return {"kind": "call", "name": "order_status", "args": {"order_id": m.group()}}
            return {"kind": "say", "store": True, "text": "Sure. What's your order number?"}
        if "callback" in lower or "call me" in lower or "call back" in lower:
            m = re.search(r"\bfor (.+?)[.?!]*$", text, flags=re.IGNORECASE)
            slot = m.group(1) if m else "the next available slot"
            return {"kind": "call", "name": "book_callback", "args": {"preferred_time": slot}}
        return {
            "kind": "say",
            "store": True,
            "text": "I can check an order, book a callback, or connect you to a person. "
            "What would you like?",
        }

    def _speak_result(self, output_item: dict[str, Any]) -> str:
        call = next(
            (
                i
                for i in self.items
                if i.get("type") == "function_call"
                and i.get("call_id") == output_item.get("call_id")
            ),
            {},
        )
        try:
            result = json.loads(output_item.get("output") or "{}")
        except json.JSONDecodeError:
            result = {}
        name = call.get("name")
        if "error" in result:
            return "Sorry, I couldn't complete that just now."
        if name == "order_status":
            if not result.get("found"):
                return f"I couldn't find order {result.get('order_id')}. Could you double-check it?"
            return f"Order {result['order_id']} is {result['status']}, {result['eta']}."
        if name == "book_callback":
            return (
                f"Done. You're booked for a callback {result.get('slot')}. "
                f"Your reference is {result.get('reference')}."
            )
        return "Okay, that's done."

    async def _run_response(self, resp_id: str, params: dict[str, Any]) -> None:
        metadata = params.get("metadata")
        status = "completed"
        self._emit(
            {
                "type": "response.created",
                "response": {"id": resp_id, "status": "in_progress", "metadata": metadata},
            }
        )
        try:
            await self._sleep(self.think_ms)
            plan = self._plan(params)
            if plan["kind"] == "call":
                await self._stream_call(resp_id, plan["name"], plan["args"])
            else:
                await self._stream_speech(resp_id, plan["text"], store=plan["store"])
        except asyncio.CancelledError:
            status = "cancelled"
        finally:
            self._emit(
                {
                    "type": "response.done",
                    "response": {"id": resp_id, "status": status, "metadata": metadata},
                }
            )

    async def _stream_call(self, resp_id: str, name: str, args: dict[str, Any]) -> None:
        item = {
            "id": _id("item"),
            "type": "function_call",
            "call_id": _id("call"),
            "name": name,
            "arguments": json.dumps(args),
        }
        base = {"response_id": resp_id, "output_index": 0}
        self._emit(
            {"type": "response.output_item.added", **base, "item": {**item, "arguments": ""}}
        )
        self._emit(
            {
                "type": "response.function_call_arguments.delta",
                **base,
                "item_id": item["id"],
                "call_id": item["call_id"],
                "delta": item["arguments"],
            }
        )
        self._emit(
            {
                "type": "response.function_call_arguments.done",
                **base,
                "item_id": item["id"],
                "call_id": item["call_id"],
                "name": name,
                "arguments": item["arguments"],
            }
        )
        self.items.append(item)
        self._emit({"type": "response.output_item.done", **base, "item": item})

    async def _stream_speech(self, resp_id: str, text: str, *, store: bool) -> None:
        item_id = _id("item")
        base = {"response_id": resp_id, "output_index": 0}
        item = {"id": item_id, "type": "message", "role": "assistant"}
        self._emit({"type": "response.output_item.added", **base, "item": item})
        words = text.split()
        total_ms = max(400, self.ms_per_word * len(words))
        pcm = tone(total_ms, freq_hz=200 if store else 330)
        chunk_bytes = int(24_000 * 2 * self.chunk_ms / 1000)
        chunks = [pcm[i : i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]
        sent_words = 0
        try:
            for n, chunk in enumerate(chunks, start=1):
                self._emit(
                    {
                        "type": "response.audio.delta",
                        **base,
                        "item_id": item_id,
                        "content_index": 0,
                        "delta": to_b64(chunk),
                    }
                )
                upto = round(len(words) * n / len(chunks))
                if upto > sent_words:
                    delta = " ".join(words[sent_words:upto]) + ("" if upto == len(words) else " ")
                    self._emit(
                        {
                            "type": "response.audio_transcript.delta",
                            **base,
                            "item_id": item_id,
                            "content_index": 0,
                            "delta": delta,
                        }
                    )
                    sent_words = upto
                await self._sleep(self.chunk_ms * self.stream_ratio)
        finally:
            if store:
                self.items.append({**item, "text": text})
        self._emit({"type": "response.audio.done", **base, "item_id": item_id, "content_index": 0})
        self._emit(
            {
                "type": "response.audio_transcript.done",
                **base,
                "item_id": item_id,
                "content_index": 0,
                "transcript": text,
            }
        )
        self._emit({"type": "response.output_item.done", **base, "item": item})
