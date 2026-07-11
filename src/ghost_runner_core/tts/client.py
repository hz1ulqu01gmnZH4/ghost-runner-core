"""Async streaming client for fish-speech's /v1/tts endpoint.

This slice turns one text response into raw pcm_s16le chunks for the voice
pipeline. fish-speech serializes work on the target GPU, so requests are
serialized here too — a second synthesis waits instead of competing for the
same model.

Wire format, verified against the running server: a NON-streaming wav response
carries a complete RIFF header, but the STREAMING response is headerless raw
pcm_s16le — fish-speech generates a header chunk and then drops it on the
floor (`inference_async` filters on `isinstance(chunk, bytes)` and the header
is wrapped in a numpy array). The codec parameters therefore come from a
one-time non-streaming probe at startup (probe_format), which fails fast if
the server's audio is anything but PCM16 mono; every streamed byte after that
is audio.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from collections.abc import AsyncIterator

import httpx

log = logging.getLogger(__name__)

# Shortest natural JA utterance that reliably synthesizes; the probe's audio is
# discarded — only its header matters.
_PROBE_TEXT = "はい。"


class TtsError(Exception):
    """fish-speech failed or returned invalid audio. Always surfaced, never retried invisibly."""


class _WavHeaderParser:
    """Incrementally strip and validate a streaming RIFF/WAVE header."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._header_complete = False
        self._sample_rate: int | None = None

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            raise TtsError("sample rate unavailable before complete WAV header")
        return self._sample_rate

    @property
    def header_complete(self) -> bool:
        return self._header_complete

    def feed(self, chunk: bytes) -> bytes:
        """Consume arbitrary response bytes and return only bytes after the data header."""
        if self._header_complete:
            return chunk

        self._buffer.extend(chunk)
        parsed = self._find_pcm_start()
        if parsed is None:
            return b""

        pcm_start, sample_rate = parsed
        pcm = bytes(self._buffer[pcm_start:])
        self._buffer.clear()
        self._sample_rate = sample_rate
        self._header_complete = True
        return pcm

    def finish(self) -> None:
        """Reject a stream that ended before its complete data-chunk header."""
        if not self._header_complete:
            raise TtsError("fish-speech stream ended before a complete WAV header")

    def _find_pcm_start(self) -> tuple[int, int] | None:
        data = self._buffer
        if len(data) >= 4 and data[0:4] != b"RIFF":
            raise TtsError("invalid RIFF magic in fish-speech WAV header")
        if len(data) < 12:
            return None
        if data[8:12] != b"WAVE":
            raise TtsError("invalid WAVE magic in fish-speech WAV header")

        offset = 12
        sample_rate: int | None = None
        while True:
            if len(data) < offset + 8:
                return None

            chunk_id = data[offset:offset + 4]
            chunk_size = int.from_bytes(data[offset + 4:offset + 8], "little")
            payload_start = offset + 8

            # The data size is a streaming placeholder, so its payload begins
            # immediately and must never be skipped using the declared length.
            if chunk_id == b"data":
                if sample_rate is None:
                    raise TtsError("WAV fmt chunk missing before data chunk")
                return payload_start, sample_rate

            if chunk_id == b"fmt " and chunk_size < 16:
                raise TtsError(f"invalid fmt chunk size: {chunk_size}")

            payload_end = payload_start + chunk_size
            padded_end = payload_end + (chunk_size & 1)
            if len(data) < padded_end:
                return None

            if chunk_id == b"fmt ":
                audio_format, channels, parsed_rate = struct.unpack_from(
                    "<HHI", data, payload_start)
                bits_per_sample = struct.unpack_from("<H", data, payload_start + 14)[0]
                if audio_format != 1:
                    raise TtsError(f"invalid audio_format in WAV header: {audio_format}")
                if bits_per_sample != 16:
                    raise TtsError(
                        f"invalid bits_per_sample in WAV header: {bits_per_sample}")
                if channels != 1:
                    raise TtsError(f"invalid channels in WAV header: {channels}")
                sample_rate = parsed_rate

            offset = padded_end


class TtsClient:
    def __init__(
        self,
        server_url: str,
        *,
        chunk_length: int = 200,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if (
            isinstance(chunk_length, bool)
            or not isinstance(chunk_length, int)
            or not 100 <= chunk_length <= 1000
        ):
            raise ValueError("chunk_length must be between 100 and 1000")
        self._chunk_length = chunk_length
        # A JA sentence runs ~1-3 s on the target GPU. 60 s is the hung-server
        # ceiling; the pipeline above owns turn-level cancellation, while the
        # shorter connect cap fails fast when the local server is absent.
        timeout = httpx.Timeout(60.0, connect=5.0)
        self._client = httpx.AsyncClient(
            base_url=server_url, timeout=timeout, transport=transport)
        # fish-speech serializes on the GPU anyway, so concurrent requests only
        # add contention and can interleave lifecycle state at the server.
        self._lock = asyncio.Lock()
        self._sample_rate: int | None = None

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            raise TtsError("sample rate unknown before probe_format()")
        return self._sample_rate

    async def check_reachable(self) -> None:
        """Startup fail-fast (§A9): fish-speech must answer, or core refuses to start."""
        try:
            resp = await self._client.get("/v1/health")
        except httpx.HTTPError as exc:
            raise TtsError(
                f"fish-speech unreachable at {self._client.base_url}: {exc}") from exc
        if resp.status_code != 200:
            raise TtsError(
                f"fish-speech /v1/health returned {resp.status_code} "
                f"at {self._client.base_url}")

    async def probe_format(self) -> None:
        """Pin the server's codec parameters with one tiny NON-streaming synth
        (the only response that carries a RIFF header). Startup fail-fast:
        anything but PCM16 mono refuses to start. The streamed audio afterwards
        is headerless, so a mid-run server swap to a different rate is not
        detectable — restart the core when the TTS server changes."""
        async with self._lock:
            try:
                resp = await self._client.post("/v1/tts", json={
                    "text": _PROBE_TEXT,
                    "format": "wav",
                    "streaming": False,
                    "chunk_length": self._chunk_length,
                    "max_new_tokens": 128,
                })
            except httpx.HTTPError as exc:
                raise TtsError(f"fish-speech format probe failed: {exc}") from exc
            if resp.status_code != 200:
                raise TtsError(
                    f"fish-speech format probe returned {resp.status_code}: "
                    f"{resp.text[:200]}")
            parser = _WavHeaderParser()
            parser.feed(resp.content)
            parser.finish()  # TtsError if the response was not a complete WAV
            self._sample_rate = parser.sample_rate

    async def synth_stream(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text and yield sample-aligned raw pcm_s16le chunks."""
        if not text:
            raise ValueError("text must not be empty")
        if self._sample_rate is None:
            raise TtsError("synth_stream before probe_format(); startup order bug")

        request_body = {
            "text": text,
            "format": "wav",
            "streaming": True,
            "chunk_length": self._chunk_length,
            "latency": "balanced",
            "max_new_tokens": 1024,
        }
        async with self._lock:
            carry = b""
            got_audio = False
            try:
                async with self._client.stream(
                    "POST", "/v1/tts", json=request_body) as resp:
                    if resp.status_code != 200:
                        await resp.aread()
                        raise TtsError(
                            f"fish-speech /v1/tts returned {resp.status_code}: "
                            f"{resp.text[:200]}")

                    async for response_chunk in resp.aiter_bytes():
                        aligned = carry + response_chunk
                        aligned_length = len(aligned) - (len(aligned) & 1)
                        if aligned_length:
                            got_audio = True
                            yield aligned[:aligned_length]
                        carry = aligned[aligned_length:]
            except httpx.HTTPError as exc:
                raise TtsError(f"fish-speech request failed: {exc}") from exc

            if carry:
                raise TtsError("fish-speech returned an odd trailing PCM byte")
            if not got_audio:
                raise TtsError(f"fish-speech returned no audio for {text!r}")

    async def aclose(self) -> None:
        await self._client.aclose()
