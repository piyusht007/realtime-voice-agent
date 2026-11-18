"""WebSocket client for OpenAI Realtime or Azure OpenAI Realtime."""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlencode, urlparse

import websockets
from websockets.exceptions import ConnectionClosed

from voice_agent.config import Settings

logger = logging.getLogger(__name__)


def build_url_and_headers(settings: Settings) -> tuple[str, dict[str, str]]:
    if settings.provider == "openai":
        url = f"{settings.openai_base_url}?{urlencode({'model': settings.openai_model})}"
        headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
        if settings.openai_beta_header:
            headers["OpenAI-Beta"] = settings.openai_beta_header
        return url, headers
    if settings.provider == "azure":
        host = urlparse(settings.azure_endpoint).netloc or settings.azure_endpoint
        host = host.rstrip("/")
        query = urlencode(
            {"api-version": settings.azure_api_version, "deployment": settings.azure_deployment}
        )
        return f"wss://{host}/openai/realtime?{query}", {"api-key": settings.azure_api_key}
    raise ValueError(f"no websocket URL for provider {settings.provider!r}")


class OpenAIRealtimeConnection:
    def __init__(self, ws: Any):
        self._ws = ws

    @classmethod
    async def connect(cls, settings: Settings) -> OpenAIRealtimeConnection:
        url, headers = build_url_and_headers(settings)
        logger.info("connecting to realtime endpoint %s", url.split("?")[0])
        ws = await websockets.connect(url, additional_headers=headers, max_size=None)
        return cls(ws)

    async def send(self, event: dict[str, Any]) -> None:
        try:
            await self._ws.send(json.dumps(event))
        except ConnectionClosed:
            logger.debug("send after close dropped: %s", event.get("type"))

    async def recv(self) -> dict[str, Any] | None:
        try:
            raw = await self._ws.recv()
        except ConnectionClosed:
            return None
        return json.loads(raw)

    async def close(self) -> None:
        await self._ws.close()
