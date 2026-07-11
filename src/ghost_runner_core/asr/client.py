"""Async client for whisper.cpp's whisper-server /inference endpoint.

First slice of the M1 voice loop: one push-to-talk utterance = one POST.
whisper-server serializes inference behind a global mutex, so requests are
serialized here too — a second utterance waits instead of tripping over the
server's lock.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

log = logging.getLogger(__name__)


class AsrError(Exception):
    """whisper-server unreachable or returned an error. Always surfaced, never retried invisibly."""


class WhisperClient:
    def __init__(self, server_url: str, language: str,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._language = language
        # 20 s cap: the command dispatch loop awaits transcribe() inline, so a
        # hung whisper-server must fail before the client's heartbeat gives up
        # on the link (2 missed pings × 15 s). CPU large-v3-turbo runs ~1-2 s.
        self._client = httpx.AsyncClient(base_url=server_url, timeout=20.0,
                                         transport=transport)
        self._lock = asyncio.Lock()

    async def check_reachable(self) -> None:
        """Startup fail-fast (§A9): whisper-server must answer, or core refuses to start."""
        try:
            resp = await self._client.get("/health")
        except httpx.HTTPError as exc:
            raise AsrError(
                f"whisper-server unreachable at {self._client.base_url}: {exc}") from exc
        if resp.status_code != 200:
            raise AsrError(
                f"whisper-server /health returned {resp.status_code} at {self._client.base_url}")

    async def transcribe(self, wav: bytes) -> str:
        """Transcribe one WAV utterance. Raises AsrError on any failure."""
        files = {"file": ("audio.wav", wav, "audio/wav")}
        data = {"response_format": "json", "language": self._language, "temperature": "0.0"}
        async with self._lock:
            try:
                resp = await self._client.post("/inference", files=files, data=data)
            except httpx.HTTPError as exc:
                raise AsrError(f"whisper-server request failed: {exc}") from exc
        if resp.status_code != 200:
            raise AsrError(
                f"whisper-server /inference returned {resp.status_code}: {resp.text[:200]}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise AsrError(f"whisper-server returned non-JSON: {resp.text[:200]}") from exc
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str):
            raise AsrError(f"whisper-server response has no text field: {str(body)[:200]}")
        return text.strip()

    async def aclose(self) -> None:
        await self._client.aclose()
