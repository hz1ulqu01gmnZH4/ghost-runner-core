"""Async client for Irodori-TTS-Server's OpenAI-compatible /v1/audio/speech.

Same duck type as TtsClient (fish-speech) — sample_rate / check_reachable /
probe_format / synth_stream / aclose — selected via config [tts] engine.

Wire format, verified against the running server (Irodori-TTS-500M-v3, MIT):
`response_format:"wav"` returns a complete RIFF file; `response_format:"pcm"`
returns headerless raw pcm_s16le at the model's native 48 kHz mono. The server
does not stream — each request returns one complete response — but the voice
pipeline synthesizes per sentence anyway, and v3's duration predictor makes
that fast (measured ~0.5 s for a short sentence, RTF ≈ 0.06 on long ones).
probe_format() pins the rate from one tiny wav response instead of trusting
the model card.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import httpx

from .client import _PROBE_TEXT, TtsError, _WavHeaderParser

log = logging.getLogger(__name__)


class IrodoriClient:
    def __init__(
        self,
        server_url: str,
        *,
        voice: str = "none",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not voice:
            raise ValueError("voice must not be empty (the server default is 'none')")
        self._voice = voice
        # A sentence synthesizes in well under a second (v3 duration predictor);
        # 60 s is the hung-server ceiling, turn-level cancellation lives above.
        timeout = httpx.Timeout(60.0, connect=5.0)
        self._client = httpx.AsyncClient(
            base_url=server_url, timeout=timeout, transport=transport)
        # The server serializes synthesis (IRODORI_MAX_CONCURRENT_SYNTHESIS=1);
        # queueing here keeps request lifecycles from interleaving.
        self._lock = asyncio.Lock()
        self._sample_rate: int | None = None

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            raise TtsError("sample rate unknown before probe_format()")
        return self._sample_rate

    async def check_reachable(self) -> None:
        """Startup fail-fast (§A9): the TTS server must answer, or core refuses to start."""
        try:
            resp = await self._client.get("/health")
        except httpx.HTTPError as exc:
            raise TtsError(
                f"irodori-tts unreachable at {self._client.base_url}: {exc}") from exc
        if resp.status_code != 200:
            raise TtsError(
                f"irodori-tts /health returned {resp.status_code} "
                f"at {self._client.base_url}")

    async def probe_format(self) -> None:
        """Pin the server's codec parameters from one tiny wav response.
        Anything but PCM16 mono refuses to start; the pcm responses afterwards
        are headerless, so a mid-run server swap is undetectable — restart the
        core when the TTS server changes."""
        async with self._lock:
            try:
                resp = await self._client.post(
                    "/v1/audio/speech", json=self._body(_PROBE_TEXT, "wav"))
            except httpx.HTTPError as exc:
                raise TtsError(f"irodori-tts format probe failed: {exc}") from exc
            if resp.status_code != 200:
                raise TtsError(
                    f"irodori-tts format probe returned {resp.status_code}: "
                    f"{resp.text[:200]}")
            parser = _WavHeaderParser()
            parser.feed(resp.content)
            parser.finish()  # TtsError if the response was not a complete WAV
            self._sample_rate = parser.sample_rate

    async def synth_stream(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize one sentence; yield sample-aligned raw pcm_s16le chunks
        as the response body downloads."""
        if not text:
            raise ValueError("text must not be empty")
        if self._sample_rate is None:
            raise TtsError("synth_stream before probe_format(); startup order bug")

        async with self._lock:
            carry = b""
            got_audio = False
            try:
                async with self._client.stream(
                    "POST", "/v1/audio/speech", json=self._body(text, "pcm")) as resp:
                    if resp.status_code != 200:
                        await resp.aread()
                        raise TtsError(
                            f"irodori-tts /v1/audio/speech returned {resp.status_code}: "
                            f"{resp.text[:200]}")
                    async for response_chunk in resp.aiter_bytes():
                        aligned = carry + response_chunk
                        aligned_length = len(aligned) - (len(aligned) & 1)
                        if aligned_length:
                            got_audio = True
                            yield aligned[:aligned_length]
                        carry = aligned[aligned_length:]
            except httpx.HTTPError as exc:
                raise TtsError(f"irodori-tts request failed: {exc}") from exc

            if carry:
                raise TtsError("irodori-tts returned an odd trailing PCM byte")
            if not got_audio:
                raise TtsError(f"irodori-tts returned no audio for {text!r}")

    def _body(self, text: str, response_format: str) -> dict:
        return {
            "model": "irodori-tts",
            "input": text,
            "voice": self._voice,
            "response_format": response_format,
        }

    async def aclose(self) -> None:
        await self._client.aclose()
