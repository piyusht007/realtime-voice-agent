"""Handoff summary construction and a mock human-agent queue."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

NEGATIVE = {
    "angry",
    "annoyed",
    "awful",
    "bad",
    "broken",
    "cancel",
    "complaint",
    "disappointed",
    "frustrated",
    "furious",
    "hate",
    "late",
    "lost",
    "never",
    "ridiculous",
    "refund",
    "terrible",
    "unacceptable",
    "upset",
    "useless",
    "worst",
    "wrong",
}
POSITIVE = {
    "appreciate",
    "awesome",
    "glad",
    "good",
    "great",
    "happy",
    "helpful",
    "love",
    "nice",
    "perfect",
    "thank",
    "thanks",
    "wonderful",
}

INTENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("billing_or_refund", ("refund", "charge", "charged", "billing", "invoice", "payment")),
    ("order_status", ("order", "package", "delivery", "shipped", "tracking", "where is")),
    ("callback_request", ("callback", "call me", "call back")),
    ("complaint", ("complaint", "unacceptable", "terrible", "worst", "manager")),
]

_WORD = re.compile(r"[a-z']+")


@dataclass
class TranscriptEntry:
    role: str
    text: str
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def score_sentiment(texts: list[str]) -> dict[str, Any]:
    words = [w for t in texts for w in _WORD.findall(t.lower())]
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    score = 0.0 if pos + neg == 0 else round((pos - neg) / (pos + neg), 2)
    label = "negative" if score < -0.2 else "positive" if score > 0.2 else "neutral"
    return {"label": label, "score": score, "positive_hits": pos, "negative_hits": neg}


def classify_intent(texts: list[str], tools_used: list[str], reason: str = "") -> str:
    haystack = " ".join([reason, *texts]).lower()
    for intent, keywords in INTENT_RULES:
        if any(k in haystack for k in keywords):
            return intent
    if "order_status" in tools_used:
        return "order_status"
    if "book_callback" in tools_used:
        return "callback_request"
    return "general_support"


def build_handoff_summary(
    session_id: str,
    transcript: list[TranscriptEntry],
    tools_used: list[str],
    reason: str,
    turn_latencies_ms: list[float],
) -> dict[str, Any]:
    user_texts = [e.text for e in transcript if e.role == "user"]
    latencies = [x for x in turn_latencies_ms if x is not None]
    return {
        "ticket_id": "HO-" + uuid.uuid4().hex[:8].upper(),
        "session_id": session_id,
        "created_at": datetime.now(UTC).isoformat(),
        "reason": reason,
        "intent": classify_intent(user_texts, tools_used, reason),
        "sentiment": score_sentiment(user_texts),
        "tools_used": tools_used,
        "turns": len(user_texts),
        "avg_first_audio_latency_ms": round(sum(latencies) / len(latencies), 1)
        if latencies
        else None,
        "transcript": [asdict(e) for e in transcript],
    }


class HandoffQueue:
    """Stand-in for a contact-center routing queue. In-memory, optionally mirrored to JSONL."""

    def __init__(self, path: str | None = None, maxlen: int = 100):
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._recent: list[dict[str, Any]] = []
        self._maxlen = maxlen
        self._path = Path(path) if path else None

    async def publish(self, summary: dict[str, Any]) -> None:
        await self._queue.put(summary)
        self._recent = [*self._recent, summary][-self._maxlen :]
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary) + "\n")

    async def get(self) -> dict[str, Any]:
        return await self._queue.get()

    def recent(self) -> list[dict[str, Any]]:
        return list(self._recent)

    def qsize(self) -> int:
        return self._queue.qsize()
