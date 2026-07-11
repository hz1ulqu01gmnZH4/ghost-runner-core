"""TtsClient unit tests against httpx MockTransport and the WAV probe parser.

Wire-format ground truth (verified against the live server): the NON-streaming
response is a complete WAV; the STREAMING response is headerless raw pcm_s16le.
probe_format() pins the codec from the former; synth_stream() trusts the pin.
"""

import json
import struct

import httpx
import pytest

from ghost_runner_core.tts.client import TtsClient, TtsError, _WavHeaderParser


def make_wav_header(
    *,
    sample_rate: int = 44100,
    audio_format: int = 1,
    bits_per_sample: int = 16,
    channels: int = 1,
    extra_chunks: tuple[tuple[bytes, bytes], ...] = (),
) -> bytes:
    chunks = []
    for chunk_id, payload in extra_chunks:
        chunk = chunk_id + struct.pack("<I", len(payload)) + payload
        chunks.append(chunk + (b"\x00" if len(payload) & 1 else b""))

    block_align = channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align
    fmt = struct.pack(
        "<HHIIHH",
        audio_format,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    chunks.append(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
    chunks.append(b"data" + struct.pack("<I", 0xFFFFFFFF))
    return b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + b"".join(chunks)


def probe_then_stream_handler(pcm: bytes, *, probe_rate: int = 44100, seen: dict | None = None):
    """The real server's split personality: WAV for streaming=False, raw PCM
    for streaming=True."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if seen is not None:
            seen.setdefault("bodies", []).append(body)
            seen["path"] = request.url.path
        if body["streaming"]:
            return httpx.Response(200, content=pcm,
                                  headers={"content-type": "audio/wav"})
        return httpx.Response(200, content=make_wav_header(sample_rate=probe_rate) + b"\x00\x00",
                              headers={"content-type": "audio/wav"})

    return handler


def make_client(handler, *, chunk_length: int = 200) -> TtsClient:
    return TtsClient(
        "http://tts.test",
        chunk_length=chunk_length,
        transport=httpx.MockTransport(handler),
    )


async def collect(client: TtsClient, text: str = "こんにちは") -> bytes:
    return b"".join([chunk async for chunk in client.synth_stream(text)])


# -- probe_format --------------------------------------------------------------


async def test_sample_rate_unknown_before_probe():
    client = make_client(lambda request: httpx.Response(500))
    try:
        with pytest.raises(TtsError, match="sample rate unknown before probe_format"):
            _ = client.sample_rate
    finally:
        await client.aclose()


async def test_probe_pins_sample_rate_via_non_streaming_wav():
    seen: dict = {}
    client = make_client(probe_then_stream_handler(b"", probe_rate=24000, seen=seen))
    try:
        await client.probe_format()
        assert client.sample_rate == 24000
    finally:
        await client.aclose()
    probe_body = seen["bodies"][0]
    assert probe_body["streaming"] is False and probe_body["format"] == "wav"


async def test_probe_rejects_non_wav_response():
    client = make_client(lambda request: httpx.Response(200, content=b"\x01\x00\x02\x00"))
    try:
        with pytest.raises(TtsError, match="RIFF magic"):
            await client.probe_format()
    finally:
        await client.aclose()


async def test_probe_rejects_truncated_header():
    client = make_client(lambda request: httpx.Response(200, content=b"RIFF\x00"))
    try:
        with pytest.raises(TtsError, match="complete WAV header"):
            await client.probe_format()
    finally:
        await client.aclose()


async def test_probe_non_200_includes_status_and_body():
    client = make_client(lambda request: httpx.Response(503, text="GPU unavailable"))
    try:
        with pytest.raises(TtsError, match="503.*GPU unavailable"):
            await client.probe_format()
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    ("header", "field"),
    [
        (b"RIFX" + make_wav_header()[4:], "RIFF magic"),
        (make_wav_header(audio_format=3), "audio_format"),
        (make_wav_header(bits_per_sample=24), "bits_per_sample"),
        (make_wav_header(channels=2), "channels"),
    ],
)
def test_wav_header_parser_rejects_invalid_fields(header: bytes, field: str):
    parser = _WavHeaderParser()
    with pytest.raises(TtsError, match=field):
        parser.feed(header)


def test_wav_header_parser_handles_seven_byte_boundaries():
    pcm = b"\x01\x00\x02\x00\x03\x00"
    wire = make_wav_header() + pcm
    parser = _WavHeaderParser()

    output = [parser.feed(wire[offset:offset + 7]) for offset in range(0, len(wire), 7)]
    parser.finish()

    assert b"".join(output) == pcm
    assert parser.sample_rate == 44100


def test_wav_header_parser_walks_extra_riff_subchunk():
    pcm = b"\x10\x00\x20\x00"
    header = make_wav_header(extra_chunks=((b"LIST", b"meta!"),))
    parser = _WavHeaderParser()

    assert parser.feed(header + pcm) == pcm
    parser.finish()
    assert parser.sample_rate == 44100


# -- synth_stream ----------------------------------------------------------------


async def test_synth_stream_posts_json_and_yields_raw_pcm():
    pcm = b"\x01\x00\xfe\xff\x45\x00"  # headerless, like the real server
    seen: dict = {}
    client = make_client(probe_then_stream_handler(pcm, seen=seen), chunk_length=320)
    try:
        await client.probe_format()
        assert await collect(client, "明日の天気を教えて") == pcm
        assert client.sample_rate == 44100
    finally:
        await client.aclose()

    assert seen["path"] == "/v1/tts"
    assert seen["bodies"][-1] == {
        "text": "明日の天気を教えて",
        "format": "wav",
        "streaming": True,
        "chunk_length": 320,
        "latency": "balanced",
        "max_new_tokens": 1024,
    }


async def test_synth_stream_before_probe_is_a_startup_order_bug():
    client = make_client(probe_then_stream_handler(b"\x00\x00"))
    try:
        with pytest.raises(TtsError, match="before probe_format"):
            await collect(client)
    finally:
        await client.aclose()


async def test_synth_stream_non_200_includes_status_and_body():
    def handler(request: httpx.Request) -> httpx.Response:
        if not json.loads(request.content)["streaming"]:
            return httpx.Response(200, content=make_wav_header() + b"\x00\x00")
        return httpx.Response(503, text="GPU unavailable")

    client = make_client(handler)
    try:
        await client.probe_format()
        with pytest.raises(TtsError, match="503.*GPU unavailable"):
            await collect(client)
    finally:
        await client.aclose()


async def test_synth_stream_connect_failure_raises():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, content=make_wav_header() + b"\x00\x00")
        raise httpx.ConnectError("refused", request=request)

    client = make_client(handler)
    try:
        await client.probe_format()
        with pytest.raises(TtsError, match="request failed.*refused"):
            await collect(client)
    finally:
        await client.aclose()


async def test_synth_stream_rejects_odd_total_pcm_bytes():
    client = make_client(probe_then_stream_handler(b"\x01\x00\x02"))
    try:
        await client.probe_format()
        with pytest.raises(TtsError, match="odd trailing PCM byte"):
            await collect(client)
    finally:
        await client.aclose()


async def test_synth_stream_rejects_empty_audio():
    client = make_client(probe_then_stream_handler(b""))
    try:
        await client.probe_format()
        with pytest.raises(TtsError, match="no audio"):
            await collect(client)
    finally:
        await client.aclose()


async def test_synth_stream_empty_text_makes_no_request():
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, content=make_wav_header())

    client = make_client(handler)
    try:
        with pytest.raises(ValueError, match="text must not be empty"):
            await collect(client, "")
    finally:
        await client.aclose()
    assert called is False


# -- check_reachable -------------------------------------------------------------


async def test_check_reachable_uses_v1_health():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200)

    client = make_client(handler)
    try:
        await client.check_reachable()
    finally:
        await client.aclose()
    assert seen["path"] == "/v1/health"


async def test_check_reachable_non_200_raises():
    client = make_client(lambda request: httpx.Response(404))
    try:
        with pytest.raises(TtsError, match="404"):
            await client.check_reachable()
    finally:
        await client.aclose()
