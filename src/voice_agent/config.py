"""Environment-driven configuration. Secrets are only ever read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

Provider = Literal["mock", "openai", "azure"]


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    provider: Provider = "mock"

    openai_api_key: str = ""
    openai_model: str = "gpt-realtime"
    openai_base_url: str = "wss://api.openai.com/v1/realtime"
    openai_beta_header: str = "realtime=v1"

    azure_endpoint: str = ""
    azure_api_key: str = ""
    azure_deployment: str = ""
    azure_api_version: str = "2024-10-01-preview"

    voice: str = "alloy"
    transcription_model: str = "whisper-1"

    vad_threshold: float = 0.5
    vad_prefix_padding_ms: int = 300
    vad_silence_ms: int = 500

    filler_threshold_ms: int = 700
    filler_text: str = "One moment while I check that for you."
    handoff_farewell_timeout_s: float = 8.0

    mock_order_latency_ms: int = 1200
    mock_callback_latency_ms: int = 250
    mock_speed: float = 1.0
    mock_think_ms: int = 250

    handoff_queue_path: str = ""
    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_env(cls) -> Settings:
        provider = _env("REALTIME_PROVIDER", "mock").lower()
        if provider not in ("mock", "openai", "azure"):
            raise ValueError(f"REALTIME_PROVIDER must be mock|openai|azure, got {provider!r}")
        return cls(
            provider=provider,  # type: ignore[arg-type]
            openai_api_key=_env("OPENAI_API_KEY"),
            openai_model=_env("OPENAI_REALTIME_MODEL", cls.openai_model),
            openai_base_url=_env("OPENAI_REALTIME_URL", cls.openai_base_url),
            openai_beta_header=_env("OPENAI_BETA_HEADER", cls.openai_beta_header),
            azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
            azure_api_key=_env("AZURE_OPENAI_API_KEY"),
            azure_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
            azure_api_version=_env("AZURE_OPENAI_API_VERSION", cls.azure_api_version),
            voice=_env("VOICE", cls.voice),
            transcription_model=_env("TRANSCRIPTION_MODEL", cls.transcription_model),
            vad_threshold=_env_float("VAD_THRESHOLD", cls.vad_threshold),
            vad_prefix_padding_ms=_env_int("VAD_PREFIX_PADDING_MS", cls.vad_prefix_padding_ms),
            vad_silence_ms=_env_int("VAD_SILENCE_MS", cls.vad_silence_ms),
            filler_threshold_ms=_env_int("FILLER_THRESHOLD_MS", cls.filler_threshold_ms),
            filler_text=_env("FILLER_TEXT", cls.filler_text),
            handoff_farewell_timeout_s=_env_float(
                "HANDOFF_FAREWELL_TIMEOUT_S", cls.handoff_farewell_timeout_s
            ),
            mock_order_latency_ms=_env_int("MOCK_ORDER_LATENCY_MS", cls.mock_order_latency_ms),
            mock_callback_latency_ms=_env_int(
                "MOCK_CALLBACK_LATENCY_MS", cls.mock_callback_latency_ms
            ),
            mock_speed=_env_float("MOCK_SPEED", cls.mock_speed),
            mock_think_ms=_env_int("MOCK_THINK_MS", cls.mock_think_ms),
            handoff_queue_path=_env("HANDOFF_QUEUE_PATH"),
            log_level=_env("LOG_LEVEL", cls.log_level),
            host=_env("HOST", cls.host),
            port=_env_int("PORT", cls.port),
        )

    def validate(self) -> None:
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when REALTIME_PROVIDER=openai")
        if self.provider == "azure":
            missing = [
                n
                for n, v in (
                    ("AZURE_OPENAI_ENDPOINT", self.azure_endpoint),
                    ("AZURE_OPENAI_API_KEY", self.azure_api_key),
                    ("AZURE_OPENAI_DEPLOYMENT", self.azure_deployment),
                )
                if not v
            ]
            if missing:
                raise ValueError(f"Missing Azure settings: {', '.join(missing)}")
