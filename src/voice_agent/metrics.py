"""Per-turn latency metrics, logged as one structured JSON line per turn."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("voice_agent.metrics")


@dataclass
class ToolMetric:
    name: str
    duration_ms: float
    filler_played: bool


@dataclass
class TurnMetrics:
    session_id: str
    turn_id: int
    source: str  # "voice" | "text"
    speech_stopped_at: float
    first_audio_at: float | None = None
    first_audio_was_filler: bool = False
    first_answer_audio_at: float | None = None
    tool_calls: list[ToolMetric] = field(default_factory=list)
    barge_in: bool = False
    outcome: str = "completed"

    @property
    def first_audio_latency_ms(self) -> float | None:
        if self.first_audio_at is None:
            return None
        return round((self.first_audio_at - self.speech_stopped_at) * 1000, 1)

    @property
    def answer_latency_ms(self) -> float | None:
        if self.first_answer_audio_at is None:
            return None
        return round((self.first_answer_audio_at - self.speech_stopped_at) * 1000, 1)

    def to_record(self) -> dict[str, Any]:
        return {
            "event": "turn.metrics",
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "source": self.source,
            "first_audio_latency_ms": self.first_audio_latency_ms,
            "first_audio_was_filler": self.first_audio_was_filler,
            "answer_latency_ms": self.answer_latency_ms,
            "tool_calls": [asdict(t) for t in self.tool_calls],
            "barge_in": self.barge_in,
            "outcome": self.outcome,
        }


class MetricsSink:
    """Collects finished turn records and emits them as JSON log lines."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def emit(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        logger.info(json.dumps(record))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if msg.startswith("{"):
            return msg
        return json.dumps(
            {"level": record.levelname, "logger": record.name, "message": msg},
        )


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("voice_agent")
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    root.propagate = False
