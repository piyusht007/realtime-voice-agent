"""Small pcm16 helpers (24 kHz, mono, little-endian)."""

from __future__ import annotations

import base64
import math
import struct

SAMPLE_RATE = 24_000
BYTES_PER_SAMPLE = 2


def pcm16_duration_ms(num_bytes: int, sample_rate: int = SAMPLE_RATE) -> float:
    return num_bytes / BYTES_PER_SAMPLE / sample_rate * 1000.0


def b64_duration_ms(b64: str, sample_rate: int = SAMPLE_RATE) -> float:
    # base64 expands 3 bytes -> 4 chars; padding is accounted for exactly.
    padding = b64.count("=", max(0, len(b64) - 2))
    return pcm16_duration_ms(len(b64) * 3 // 4 - padding, sample_rate)


def rms(pcm: bytes) -> float:
    """Normalised RMS energy in [0, 1]."""
    n = len(pcm) // BYTES_PER_SAMPLE
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm[: n * BYTES_PER_SAMPLE])
    return math.sqrt(sum(s * s for s in samples) / n) / 32768.0


def tone(duration_ms: int, freq_hz: float = 220.0, amplitude: float = 0.15) -> bytes:
    """Synthesize a soft, slightly warbling tone used as placeholder 'speech' in mock mode."""
    n = int(SAMPLE_RATE * duration_ms / 1000)
    out = bytearray()
    for i in range(n):
        t = i / SAMPLE_RATE
        env = min(1.0, i / 480, (n - i) / 480)  # 20 ms fade in/out
        f = freq_hz * (1 + 0.08 * math.sin(2 * math.pi * 3 * t))
        out += struct.pack("<h", int(32767 * amplitude * env * math.sin(2 * math.pi * f * t)))
    return bytes(out)


def silence(duration_ms: int) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * duration_ms / 1000)


def to_b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


def from_b64(b64: str) -> bytes:
    return base64.b64decode(b64)
