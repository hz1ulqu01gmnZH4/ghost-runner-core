"""IrodoriClient unit tests against httpx MockTransport.

Wire-format ground truth (verified against the live Irodori-TTS-Server):
response_format "wav" → complete RIFF file; "pcm" → headerless raw pcm_s16le
48 kHz mono; GET /health for liveness. No streaming — one complete response."""

import json

import httpx
import pytest

from ghost_runner_core.config import ConfigError, load_config
from ghost_runner_core.tts.irodori import IrodoriClient
from ghost_runner_core.tts.client import TtsError

from test_tts_client import make_wav_header


def irodori_handler(pcm: bytes, *, probe_rate: int = 48000, seen: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        body = json.loads(request.content)
        if seen is not None:
            seen.setdefault("bodies", []).append(body)
            seen["path"] = request.url.path
        if body["response_format"] == "wav":
            return httpx.Response(
                200, content=make_wav_header(sample_rate=probe_rate) + b"\x00\x00")
        return httpx.Response(200, content=pcm,
                              headers={"content-type": "application/octet-stream"})
    return handler


def make_client(handler, *, voice: str = "none") -> IrodoriClient:
    return IrodoriClient("http://irodori.test", voice=voice,
                         transport=httpx.MockTransport(handler))


async def collect(client: IrodoriClient, text: str = "こんにちは") -> bytes:
    return b"".join([chunk async for chunk in client.synth_stream(text)])


async def test_probe_pins_48khz_and_synth_yields_raw_pcm():
    pcm = b"\x65\xff\x60\xff\x45\xff"  # first real bytes seen on the wire
    seen: dict = {}
    client = make_client(irodori_handler(pcm, seen=seen), voice="none")
    try:
        await client.probe_format()
        assert client.sample_rate == 48000
        assert await collect(client, "月が昇る。") == pcm
    finally:
        await client.aclose()
    assert seen["path"] == "/v1/audio/speech"
    probe, synth = seen["bodies"]
    assert probe == {"model": "irodori-tts", "input": "はい。",
                     "voice": "none", "response_format": "wav"}
    assert synth == {"model": "irodori-tts", "input": "月が昇る。",
                     "voice": "none", "response_format": "pcm"}


async def test_voice_is_forwarded():
    seen: dict = {}
    client = make_client(irodori_handler(b"\x00\x00", seen=seen), voice="mei")
    try:
        await client.probe_format()
        await collect(client)
    finally:
        await client.aclose()
    assert all(b["voice"] == "mei" for b in seen["bodies"])


def test_empty_voice_is_a_caller_bug():
    with pytest.raises(ValueError, match="voice must not be empty"):
        IrodoriClient("http://irodori.test", voice="")


async def test_sample_rate_unknown_before_probe():
    client = make_client(irodori_handler(b""))
    try:
        with pytest.raises(TtsError, match="sample rate unknown before probe_format"):
            _ = client.sample_rate
        with pytest.raises(TtsError, match="before probe_format"):
            await collect(client)
    finally:
        await client.aclose()


async def test_probe_rejects_non_wav_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x01\x00\x02\x00")
    client = make_client(handler)
    try:
        with pytest.raises(TtsError, match="RIFF magic"):
            await client.probe_format()
    finally:
        await client.aclose()


async def test_synth_error_body_is_surfaced():
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["response_format"] == "wav":
            return httpx.Response(200, content=make_wav_header() + b"\x00\x00")
        return httpx.Response(400, text='{"error":{"message":"Unknown voice"}}')
    client = make_client(handler)
    try:
        await client.probe_format()
        with pytest.raises(TtsError, match="400.*Unknown voice"):
            await collect(client)
    finally:
        await client.aclose()


async def test_odd_pcm_byte_count_fails_loud():
    client = make_client(irodori_handler(b"\x01\x00\x02"))
    try:
        await client.probe_format()
        with pytest.raises(TtsError, match="odd trailing PCM byte"):
            await collect(client)
    finally:
        await client.aclose()


async def test_empty_audio_fails_loud():
    client = make_client(irodori_handler(b""))
    try:
        await client.probe_format()
        with pytest.raises(TtsError, match="no audio"):
            await collect(client)
    finally:
        await client.aclose()


async def test_check_reachable_uses_health():
    client = make_client(irodori_handler(b""))
    try:
        await client.check_reachable()
    finally:
        await client.aclose()


async def test_check_reachable_failure_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)
    client = make_client(handler)
    try:
        with pytest.raises(TtsError, match="unreachable"):
            await client.check_reachable()
    finally:
        await client.aclose()


# -- config ------------------------------------------------------------------


BASE_CONFIG = """\
[server]
bind = "127.0.0.1"
port = 8790

[llm]
base_url = "http://127.0.0.1:8080/v1"
model = "test"

[memory]
db = "/tmp/test.db"
"""


def test_config_tts_engine_defaults_to_fish(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(BASE_CONFIG + '\n[tts]\nserver_url = "http://x"\n')
    cfg = load_config(p).tts
    assert cfg.engine == "fish" and cfg.voice == "none"


def test_config_tts_irodori_engine_with_voice(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(BASE_CONFIG
                 + '\n[tts]\nserver_url = "http://x"\nengine = "irodori"\nvoice = "mei"\n')
    cfg = load_config(p).tts
    assert cfg.engine == "irodori" and cfg.voice == "mei"


def test_config_tts_unknown_engine_fails(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(BASE_CONFIG + '\n[tts]\nserver_url = "http://x"\nengine = "espeak"\n')
    with pytest.raises(ConfigError, match="tts.engine"):
        load_config(p)


def test_config_tts_voice_requires_irodori(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(BASE_CONFIG + '\n[tts]\nserver_url = "http://x"\nvoice = "mei"\n')
    with pytest.raises(ConfigError, match="only supported by the irodori engine"):
        load_config(p)
