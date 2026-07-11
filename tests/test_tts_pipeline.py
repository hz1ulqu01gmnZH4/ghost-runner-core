"""TurnSpeech tests: sentence-queue worker, audio_meta open/close discipline,
binary frame sequencing, cancellation, and fail-loud synthesis errors.

Uses a real SessionManager over a fake socket so the assertions cover the
actual broadcast/encode path, and a fake synth client standing in for
tts.client.TtsClient (same duck type: sample_rate + synth_stream)."""

import asyncio
import json

import pytest

from ghost_runner_core.binframe import KIND_TTS_PCM, decode_frame
from ghost_runner_core.config import ConfigError, load_config
from ghost_runner_core.server.session import SessionManager
from ghost_runner_core.voice.tts_pipeline import TurnSpeech


class FakeWs:
    def __init__(self):
        self.text: list[dict] = []
        self.binary: list[bytes] = []

    async def send(self, data):
        if isinstance(data, bytes):
            self.binary.append(data)
        else:
            self.text.append(json.loads(data))


class FakeSynth:
    """Yields each sentence back as one PCM chunk per 4 chars (deterministic)."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.requests: list[str] = []
        self.block_on: str | None = None   # sentence text that hangs forever
        self.fail_on: str | None = None    # sentence text that raises

    async def synth_stream(self, text: str):
        self.requests.append(text)
        if text == self.fail_on:
            raise RuntimeError(f"synthesis exploded on {text!r}")
        if text == self.block_on:
            await asyncio.Event().wait()
        data = text.encode("utf-8")
        for i in range(0, len(data), 4):
            yield data[i:i + 4]


def make_sessions() -> tuple[SessionManager, FakeWs]:
    sessions = SessionManager()
    ws = FakeWs()
    sessions.create_session(ws)
    return sessions, ws


def metas(ws: FakeWs) -> list[dict]:
    return [e for e in ws.text if e["type"] == "audio_meta"]


async def test_two_sentences_one_stream_with_open_and_close():
    sessions, ws = make_sessions()
    speech = TurnSpeech(FakeSynth(), sessions, turn_id=7)
    speech.feed("こんにちは。")
    speech.feed("元気です")
    await speech.finish()

    open_meta, close_meta = metas(ws)
    assert open_meta["payload"] == {"stream": 2, "codec": "pcm_s16le",
                                    "sample_rate": 44100, "channels": 1, "last": False}
    assert open_meta["turn"] == 7
    assert close_meta["payload"]["last"] is True
    assert close_meta["payload"]["stream"] == 2

    payload = b""
    for i, frame in enumerate(ws.binary):
        kind, stream_id, seq, pcm = decode_frame(frame)
        assert (kind, stream_id, seq) == (KIND_TTS_PCM, 2, i)
        payload += pcm
    assert payload == "こんにちは。".encode() + "元気です".encode()


async def test_meta_precedes_first_binary_frame():
    sessions, ws = make_sessions()
    order: list[str] = []
    real_send = ws.send

    async def spy(data):
        order.append("bin" if isinstance(data, bytes) else json.loads(data)["type"])
        await real_send(data)
    ws.send = spy

    speech = TurnSpeech(FakeSynth(), sessions, turn_id=1)
    speech.feed("おはよう。")
    await speech.finish()
    assert order[0] == "audio_meta" and order[-1] == "audio_meta"
    assert set(order[1:-1]) == {"bin"}


async def test_silent_reply_opens_no_stream():
    sessions, ws = make_sessions()
    speech = TurnSpeech(FakeSynth(), sessions, turn_id=1)
    speech.feed("（…）")  # nothing speakable
    await speech.finish()
    assert metas(ws) == [] and ws.binary == []


async def test_stream_ids_are_even_and_increase_per_turn():
    sessions, _ = make_sessions()
    a = TurnSpeech(FakeSynth(), sessions, turn_id=1)
    b = TurnSpeech(FakeSynth(), sessions, turn_id=2)
    assert a._stream_id == 2 and b._stream_id == 4
    await a.cancel()
    await b.cancel()


async def test_cancel_mid_synthesis_stops_without_close_meta():
    sessions, ws = make_sessions()
    synth = FakeSynth()
    synth.block_on = "二つ目。"
    speech = TurnSpeech(synth, sessions, turn_id=3)
    # The trailing そ forces 二つ目。 out of the chunker's closer-hold.
    speech.feed("一つ目。二つ目。そ")
    # Wait until the first sentence's audio went out and the worker is stuck
    # inside the second sentence's synthesis.
    async with asyncio.timeout(5):
        while len(synth.requests) < 2 or not ws.binary:
            await asyncio.sleep(0)
    await speech.cancel()
    assert [m["payload"]["last"] for m in metas(ws)] == [False]  # opened, never closed
    assert synth.requests == ["一つ目。", "二つ目。"]


async def test_synthesis_failure_propagates_from_finish():
    sessions, _ = make_sessions()
    synth = FakeSynth()
    synth.fail_on = "壊れる。"
    speech = TurnSpeech(synth, sessions, turn_id=4)
    speech.feed("壊れる。")
    with pytest.raises(RuntimeError, match="synthesis exploded"):
        await speech.finish()


async def test_first_audio_latency_is_recorded():
    sessions, _ = make_sessions()
    speech = TurnSpeech(FakeSynth(), sessions, turn_id=5)
    assert speech.first_audio_ms is None
    speech.feed("はい。")
    await speech.finish()
    assert isinstance(speech.first_audio_ms, int) and speech.first_audio_ms >= 0


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


def test_config_without_tts_section(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(BASE_CONFIG)
    assert load_config(p).tts is None


def test_config_with_tts_section(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(BASE_CONFIG + '\n[tts]\nserver_url = "http://127.0.0.1:8930"\n')
    assert load_config(p).tts.server_url == "http://127.0.0.1:8930"


def test_config_tts_missing_url_fails(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(BASE_CONFIG + "\n[tts]\n")
    with pytest.raises(ConfigError, match="tts.server_url"):
        load_config(p)


def test_config_tts_unknown_key_fails(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(BASE_CONFIG + '\n[tts]\nserver_url = "http://x"\nspeed = 1.5\n')
    with pytest.raises(ConfigError, match="unknown config key tts.speed"):
        load_config(p)
