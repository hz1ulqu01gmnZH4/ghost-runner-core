"""A synth chunk bigger than PCM_FRAME_BYTES must be sliced into bounded
frames — hello_ack.limits.max_bin_frame is a promise, not a suggestion."""

import json

from ghost_runner_core.binframe import decode_frame
from ghost_runner_core.server.session import SessionManager
from ghost_runner_core.server.ws_server import MAX_BIN_FRAME
from ghost_runner_core.voice.tts_pipeline import PCM_FRAME_BYTES, TurnSpeech


class FakeWs:
    def __init__(self):
        self.binary: list[bytes] = []

    async def send(self, data):
        if isinstance(data, bytes):
            self.binary.append(data)
        else:
            json.loads(data)  # meta envelopes: parse, discard


class BigChunkSynth:
    sample_rate = 44100

    async def synth_stream(self, text):
        yield b"\x01\x02" * ((PCM_FRAME_BYTES * 3) // 2 + 5)  # 3.00015… frames


async def test_oversized_synth_chunk_is_sliced():
    sessions = SessionManager()
    ws = FakeWs()
    sessions.create_session(ws)
    speech = TurnSpeech(BigChunkSynth(), sessions, turn_id=1)
    speech.feed("長い。")
    await speech.finish()

    assert len(ws.binary) == 4  # 3 full slices + remainder
    total = b""
    for i, frame in enumerate(ws.binary):
        assert len(frame) <= MAX_BIN_FRAME
        _, _, seq, pcm = decode_frame(frame)
        assert seq == i
        assert len(pcm) <= PCM_FRAME_BYTES
        total += pcm
    assert total == b"\x01\x02" * ((PCM_FRAME_BYTES * 3) // 2 + 5)
