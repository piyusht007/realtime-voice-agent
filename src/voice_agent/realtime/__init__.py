"""Realtime connection factory."""

from __future__ import annotations

from voice_agent.config import Settings
from voice_agent.realtime.base import RealtimeConnection
from voice_agent.realtime.mock import MockRealtimeConnection


async def connect(settings: Settings) -> RealtimeConnection:
    if settings.provider == "mock":
        return MockRealtimeConnection(speed=settings.mock_speed, think_ms=settings.mock_think_ms)
    from voice_agent.realtime.openai_ws import OpenAIRealtimeConnection

    settings.validate()
    return await OpenAIRealtimeConnection.connect(settings)


__all__ = ["RealtimeConnection", "MockRealtimeConnection", "connect"]
