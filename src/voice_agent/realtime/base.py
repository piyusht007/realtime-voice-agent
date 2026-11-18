"""Transport abstraction over a Realtime API session (real or mock)."""

from __future__ import annotations

from typing import Any, Protocol


class RealtimeConnection(Protocol):
    """A bidirectional JSON event stream compatible with the Realtime API event protocol."""

    async def send(self, event: dict[str, Any]) -> None: ...

    async def recv(self) -> dict[str, Any] | None:
        """Next server event, or None once the connection is closed."""
        ...

    async def close(self) -> None: ...


# The beta and GA protocols use slightly different names for the same events.
AUDIO_DELTA_EVENTS = {"response.audio.delta", "response.output_audio.delta"}
AUDIO_TRANSCRIPT_DELTA_EVENTS = {
    "response.audio_transcript.delta",
    "response.output_audio_transcript.delta",
}
AUDIO_TRANSCRIPT_DONE_EVENTS = {
    "response.audio_transcript.done",
    "response.output_audio_transcript.done",
}
