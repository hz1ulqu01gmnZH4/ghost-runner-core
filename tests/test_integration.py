"""End-to-end protocol tests: real WsServer + SessionManager + Orchestrator +
TurnManager + LlmScheduler, with a fake LlamaClient. Covers the M0 kill-gate
flows: chat turn state sequence, token streaming, barge-in, resume/replay,
ownership, and protocol-violation handling."""

import asyncio
import json

import pytest
import websockets

from ghost_runner_core.asr.client import AsrError
from ghost_runner_core.binframe import KIND_TTS_PCM, decode_frame
from ghost_runner_core.config import LlmConfig
from ghost_runner_core.envelope import Envelope
from ghost_runner_core.tts.client import TtsError
from ghost_runner_core.llm.client import LlmError
from ghost_runner_core.llm.scheduler import LlmScheduler
from ghost_runner_core.orchestrator.orchestrator import Orchestrator
from ghost_runner_core.server.session import SessionManager
from ghost_runner_core.server.ws_server import WsServer
from ghost_runner_core.store import CoreStore

LLM_CFG = LlmConfig(base_url="fake", model="fake", system_prompt="テスト用。",
                    history_messages=20)


class FakeLlm:
    """Streams scripted deltas; optionally gates mid-stream for barge-in tests."""

    def __init__(self, deltas=("こん", "にちは", "!"), gate: asyncio.Event | None = None,
                 gate_after: int = 1):
        self.deltas = deltas
        self.gate = gate
        self.gate_after = gate_after
        self.calls = 0

    async def chat_stream(self, messages):
        self.calls += 1
        for i, delta in enumerate(self.deltas):
            if self.gate is not None and i == self.gate_after:
                await self.gate.wait()
            await asyncio.sleep(0)
            yield delta


class FailingLlm(FakeLlm):
    """Streams one delta, then dies — exercises the honest ERROR path."""

    async def chat_stream(self, messages):
        self.calls += 1
        yield "途中"
        await asyncio.sleep(0)
        raise LlmError("llama-server exploded (test)")


class FakeAsr:
    """Matches WhisperClient.transcribe(); records the exact bytes it received."""

    def __init__(self, text="明日の天気を教えて", error: str | None = None):
        self.text = text
        self.error = error
        self.received: list[bytes] = []

    async def transcribe(self, wav: bytes) -> str:
        self.received.append(wav)
        if self.error is not None:
            raise AsrError(self.error)
        return self.text


class Core:
    def __init__(self, llm, tmp_path, asr=None, skills=None, tts=None):
        self.store = CoreStore(tmp_path / "ghost.db")
        self.sessions = SessionManager()
        self.scheduler = LlmScheduler()
        self.orchestrator = Orchestrator(self.store, self.sessions, self.scheduler,
                                         llm, LLM_CFG, asr=asr, skills=skills, tts=tts)
        self.server = WsServer("127.0.0.1", 0, self.sessions,
                               self.orchestrator.handle_command, auth_token=None,
                               resync_streams=self.orchestrator.resync_streams)

    async def __aenter__(self):
        self.scheduler.start()
        await self.server.start()
        await self.orchestrator.startup()
        self.url = f"ws://127.0.0.1:{self.server.port}"
        return self

    async def __aexit__(self, *exc):
        await self.server.stop()
        await self.scheduler.stop()
        self.store.close()


class Client:
    """Thin protocol client: hello handshake + typed frame collection."""

    def __init__(self, url):
        self.url = url
        self.ws = None
        self.hello_ack = None
        self.frames: list[dict] = []
        self.binary: list[tuple[int, int, int, bytes]] = []  # (kind, stream, seq, payload)
        self._cmd_n = 0

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()

    async def connect(self, resume: dict | None = None):
        self.ws = await websockets.connect(self.url)
        payload = {"client": "test", "app_version": "0", "accept_v": [1],
                   "wants": ["state", "token", "turn"]}
        if resume:
            payload["resume"] = resume
        await self.ws.send(Envelope(type="hello", id="h1", payload=payload).encode())
        self.hello_ack = json.loads(await self.ws.recv())
        assert self.hello_ack["type"] == "hello_ack"

    async def command(self, name, args):
        self._cmd_n += 1
        cid = f"c{self._cmd_n}"
        await self.ws.send(Envelope(type="command", id=cid,
                                    payload={"name": name, "args": args}).encode())
        return cid

    async def recv_until(self, pred, timeout=5.0):
        """Collect frames until pred(frame) is true; returns that frame.
        Binary frames are decoded into self.binary and never match pred."""
        async with asyncio.timeout(timeout):
            while True:
                raw = await self.ws.recv()
                if isinstance(raw, bytes):
                    self.binary.append(decode_frame(raw))
                    continue
                frame = json.loads(raw)
                self.frames.append(frame)
                if pred(frame):
                    return frame

    def states(self):
        return [f["payload"]["state"] for f in self.frames if f["type"] == "state"]

    def token_text(self, stream):
        toks = sorted((f["payload"] for f in self.frames
                       if f["type"] == "token" and f["payload"]["stream"] == stream),
                      key=lambda p: p["seq"])
        return "".join(t["delta"] for t in toks)

    def max_seq(self):
        return max((f["seq"] for f in self.frames if "seq" in f), default=0)


async def test_chat_turn_full_lifecycle(tmp_path):
    async with Core(FakeLlm(), tmp_path) as core, Client(core.url) as c:
        await c.recv_until(lambda f: f["type"] == "state")  # snapshot resend: IDLE
        assert c.frames[-1]["payload"]["state"] == "IDLE"

        cid = await c.command("chat.send", {"text": "やあ"})
        ack = await c.recv_until(lambda f: f["type"] == "ack" and f["id"] == cid)
        turn = ack["payload"]["result"]["turn"]

        await c.recv_until(lambda f: f["type"] == "turn"
                           and f["payload"]["event"] == "done" and f["turn"] == turn)
        await c.recv_until(lambda f: f["type"] == "state"
                           and f["payload"]["state"] == "IDLE")
        seq = [s for s in c.states() if s != "IDLE"]
        assert seq == ["THINKING", "RESPONDING", "SUCCESS"]
        assert c.token_text(f"t{turn}") == "こんにちは!"
        done = [f for f in c.frames if f["type"] == "token" and f["payload"]["done"]]
        assert done and done[0]["payload"]["finish"] == "stop"
        # persisted conversation
        assert ("assistant", "こんにちは!") in core.store.recent_messages(10)


async def test_barge_in_supersedes_active_turn(tmp_path):
    gate = asyncio.Event()
    async with Core(FakeLlm(deltas=("あ", "い", "う"), gate=gate), tmp_path) as core, \
            Client(core.url) as c:
        cid1 = await c.command("chat.send", {"text": "一つ目"})
        ack1 = await c.recv_until(lambda f: f["type"] == "ack" and f["id"] == cid1)
        turn1 = ack1["payload"]["result"]["turn"]
        await c.recv_until(lambda f: f["type"] == "token" and f["turn"] == turn1)

        gate.set()  # unblock the fake stream; the second send races it out anyway
        cid2 = await c.command("chat.send", {"text": "二つ目"})
        cancelled = await c.recv_until(
            lambda f: f["type"] == "turn" and f["payload"]["event"] == "cancelled")
        assert cancelled["turn"] == turn1
        assert cancelled["payload"]["reason"] == "superseded"
        ack2 = await c.recv_until(lambda f: f["type"] == "ack" and f["id"] == cid2)
        turn2 = ack2["payload"]["result"]["turn"]
        await c.recv_until(lambda f: f["type"] == "turn"
                           and f["payload"]["event"] == "done" and f["turn"] == turn2)
        assert c.token_text(f"t{turn2}") == "あいう"


async def test_user_cancel_and_ownership(tmp_path):
    gate = asyncio.Event()
    async with Core(FakeLlm(gate=gate), tmp_path) as core, Client(core.url) as owner, \
            Client(core.url) as other:
        cid = await owner.command("chat.send", {"text": "長い話"})
        ack = await owner.recv_until(lambda f: f["type"] == "ack" and f["id"] == cid)
        turn = ack["payload"]["result"]["turn"]

        # A non-owner session may not cancel a balloon turn (§A7.6).
        bad = await other.command("turn.cancel", {"turn": turn})
        err = await other.recv_until(lambda f: f["type"] == "error" and f.get("id") == bad)
        assert err["payload"]["code"] == "not_owner"

        good = await owner.command("turn.cancel", {"turn": turn})
        # The cancelled broadcast precedes the ack on the wire — wait in order.
        await owner.recv_until(lambda f: f["type"] == "turn"
                               and f["payload"]["event"] == "cancelled")
        ok = await owner.recv_until(lambda f: f["type"] == "ack" and f["id"] == good)
        assert ok["payload"]["result"]["cancelled"] is True
        # idempotent second cancel
        again = await owner.command("turn.cancel", {"turn": turn})
        ok2 = await owner.recv_until(lambda f: f["type"] == "ack" and f["id"] == again)
        assert ok2["payload"]["result"]["cancelled"] is False
        gate.set()


async def test_resume_replays_missed_notifications(tmp_path):
    async with Core(FakeLlm(), tmp_path) as core:
        c = Client(core.url)
        await c.connect()
        await c.recv_until(lambda f: f["type"] == "state")
        resume_key = c.hello_ack["payload"]["resume_key"]
        original_session = c.hello_ack["payload"]["session"]
        last_seen = c.max_seq()
        await c.ws.close()

        # Activity while disconnected: a full chat turn happens (driven internally).
        await core.orchestrator.handle_command("s-internal", "chat.send", {"text": "留守中"})
        for _ in range(200):
            await asyncio.sleep(0.02)
            if core.orchestrator.turns.current is None \
                    and core.sessions.state_snapshot.payload["state"] == "IDLE":
                break

        await c.connect(resume={"resume_key": resume_key, "last_seen_seq": last_seen})
        assert c.hello_ack["payload"]["replay"] is True
        assert c.hello_ack["payload"]["session"] == original_session  # same session resumed
        idle = await c.recv_until(lambda f: f["type"] == "state"
                                  and f["payload"]["state"] == "IDLE"
                                  and f["seq"] > last_seen + 3)
        states = [f["payload"]["state"] for f in c.frames if f["type"] == "state"]
        assert "THINKING" in states and "SUCCESS" in states  # the missed turn, in order
        assert idle["seq"] == c.max_seq()
        await c.ws.close()

        # A resume key from nowhere → honest fresh session
        c2 = Client(core.url)
        await c2.connect(resume={"resume_key": "bogus", "last_seen_seq": 0})
        assert c2.hello_ack["payload"]["replay"] is False
        await c2.ws.close()


async def test_protocol_violations_close_4400(tmp_path):
    async with Core(FakeLlm(), tmp_path) as core:
        ws = await websockets.connect(core.url)
        await ws.send(Envelope(type="hello", id="h1", payload={
            "client": "t", "app_version": "0", "accept_v": [1], "wants": []}).encode())
        ack = json.loads(await ws.recv())
        assert ack["type"] == "hello_ack"
        json.loads(await ws.recv())  # snapshot state

        for _ in range(3):
            await ws.send("garbage")
            err = json.loads(await ws.recv())
            assert err["type"] == "error" and err["payload"]["code"] == "bad_envelope"
        with pytest.raises(websockets.ConnectionClosed) as closed:
            await ws.recv()
        assert closed.value.rcvd.code == 4400


async def test_unknown_command_and_wrong_version(tmp_path):
    async with Core(FakeLlm(), tmp_path) as core:
        async with Client(core.url) as c:
            cid = await c.command("nope.nope", {})
            err = await c.recv_until(lambda f: f["type"] == "error" and f.get("id") == cid)
            assert err["payload"]["code"] == "unknown_command"

        ws = await websockets.connect(core.url)
        await ws.send('{"v":9,"type":"hello","payload":{}}')
        err = json.loads(await ws.recv())
        assert err["payload"]["code"] == "unsupported_version"
        with pytest.raises(websockets.ConnectionClosed):
            await ws.recv()


async def test_stream_sync_snapshot(tmp_path):
    gate = asyncio.Event()
    async with Core(FakeLlm(deltas=("春は", "あけぼの"), gate=gate), tmp_path) as core, \
            Client(core.url) as c:
        cid = await c.command("chat.send", {"text": "枕草子"})
        ack = await c.recv_until(lambda f: f["type"] == "ack" and f["id"] == cid)
        turn = ack["payload"]["result"]["turn"]
        await c.recv_until(lambda f: f["type"] == "token" and f["turn"] == turn)

        sid = await c.command("stream.sync", {"stream": f"t{turn}"})
        snap = await c.recv_until(lambda f: f["type"] == "token"
                                  and "snapshot" in f["payload"])
        assert snap["payload"]["snapshot"]["text"] == "春は"
        await c.recv_until(lambda f: f["type"] == "ack" and f["id"] == sid)
        gate.set()
        await c.recv_until(lambda f: f["type"] == "turn" and f["payload"]["event"] == "done")


async def test_llm_failure_surfaces_as_error_state(tmp_path):
    async with Core(FailingLlm(), tmp_path) as core, Client(core.url) as c:
        cid = await c.command("chat.send", {"text": "壊れて"})
        ack = await c.recv_until(lambda f: f["type"] == "ack" and f["id"] == cid)
        turn = ack["payload"]["result"]["turn"]

        err_state = await c.recv_until(lambda f: f["type"] == "state"
                                       and f["payload"]["state"] == "ERROR")
        assert "LLM error" in err_state["payload"]["message"]
        done = await c.recv_until(lambda f: f["type"] == "turn"
                                  and f["payload"]["event"] == "done" and f["turn"] == turn)
        assert done["payload"]["reason"] == "error"
        assert core.orchestrator._stream is None  # no orphaned stream state
        # honest ERROR is transient too: the mascot recovers to IDLE
        await c.recv_until(lambda f: f["type"] == "state"
                           and f["payload"]["state"] == "IDLE")


async def test_reconnect_mid_stream_resyncs_snapshot(tmp_path):
    gate = asyncio.Event()
    async with Core(FakeLlm(deltas=("春は", "あけぼの"), gate=gate), tmp_path) as core:
        c = Client(core.url)
        await c.connect()
        cid = await c.command("chat.send", {"text": "枕草子"})
        ack = await c.recv_until(lambda f: f["type"] == "ack" and f["id"] == cid)
        turn = ack["payload"]["result"]["turn"]
        await c.recv_until(lambda f: f["type"] == "token" and f["turn"] == turn)
        resume_key = c.hello_ack["payload"]["resume_key"]
        last_seen = c.max_seq()
        await c.ws.close()

        # Reattach while the stream is still live: core must push the snapshot
        # unprompted (§A5.5 item 4), no stream.sync required.
        await c.connect(resume={"resume_key": resume_key, "last_seen_seq": last_seen})
        snap = await c.recv_until(lambda f: f["type"] == "token"
                                  and "snapshot" in f["payload"])
        assert snap["payload"]["snapshot"]["text"] == "春は"
        gate.set()
        await c.recv_until(lambda f: f["type"] == "turn" and f["payload"]["event"] == "done")


# --- asr.transcribe (M1 voice loop, first slice) -------------------------------

def _fake_wav(payload_bytes: int = 64) -> bytes:
    """Smallest thing the RIFF/WAVE guard accepts; content never reaches a real
    decoder in these tests (FakeAsr just records it)."""
    return b"RIFF" + (36 + payload_bytes).to_bytes(4, "little") + b"WAVE" \
        + b"\x00" * payload_bytes


async def test_asr_transcribe_roundtrip(tmp_path):
    import base64
    asr = FakeAsr(text="明日の天気を教えて")
    async with Core(FakeLlm(), tmp_path, asr=asr) as core, Client(core.url) as c:
        wav = _fake_wav()
        cid = await c.command("asr.transcribe", {"audio": base64.b64encode(wav).decode()})
        ack = await c.recv_until(lambda f: f["type"] == "ack" and f["id"] == cid)
        assert ack["payload"]["ok"] is True
        assert ack["payload"]["result"] == {"text": "明日の天気を教えて"}
        assert asr.received == [wav]  # decoded audio reaches ASR byte-exact


async def test_asr_transcribe_unconfigured_is_unavailable(tmp_path):
    async with Core(FakeLlm(), tmp_path) as core, Client(core.url) as c:
        cid = await c.command("asr.transcribe", {"audio": "UklGRg=="})
        err = await c.recv_until(lambda f: f["type"] == "error" and f["id"] == cid)
        assert err["payload"]["code"] == "unavailable"


async def test_asr_transcribe_rejects_bad_input(tmp_path):
    import base64
    asr = FakeAsr()
    async with Core(FakeLlm(), tmp_path, asr=asr) as core, Client(core.url) as c:
        # Not base64 at all.
        cid = await c.command("asr.transcribe", {"audio": "not base64!!"})
        err = await c.recv_until(lambda f: f["type"] == "error" and f["id"] == cid)
        assert err["payload"]["code"] == "bad_envelope"
        # Valid base64, but not a RIFF/WAVE file.
        cid = await c.command(
            "asr.transcribe", {"audio": base64.b64encode(b"\x00" * 64).decode()})
        err = await c.recv_until(lambda f: f["type"] == "error" and f["id"] == cid)
        assert err["payload"]["code"] == "bad_envelope"
        # Missing argument.
        cid = await c.command("asr.transcribe", {})
        err = await c.recv_until(lambda f: f["type"] == "error" and f["id"] == cid)
        assert err["payload"]["code"] == "bad_envelope"
        assert asr.received == []  # nothing invalid ever reached the ASR engine


async def test_asr_failure_surfaces_as_unavailable(tmp_path):
    import base64
    asr = FakeAsr(error="whisper-server exploded (test)")
    async with Core(FakeLlm(), tmp_path, asr=asr) as core, Client(core.url) as c:
        cid = await c.command(
            "asr.transcribe", {"audio": base64.b64encode(_fake_wav()).decode()})
        err = await c.recv_until(lambda f: f["type"] == "error" and f["id"] == cid)
        assert err["payload"]["code"] == "unavailable"
        assert "ASR error" in err["payload"]["message"]


async def test_asr_transcribe_accepts_multi_megabyte_audio(tmp_path):
    """A realistic PTT hold (>4 s of 44.1 kHz stereo) produces a ws frame past
    the websockets default 1 MiB max_size — the server must be sized for it
    (regression: default max_size closed the connection with 1009)."""
    import base64
    asr = FakeAsr(text="長い発話")
    async with Core(FakeLlm(), tmp_path, asr=asr) as core, Client(core.url) as c:
        wav = _fake_wav(payload_bytes=2_000_000)  # ~2.7 MB as base64
        cid = await c.command("asr.transcribe", {"audio": base64.b64encode(wav).decode()})
        ack = await c.recv_until(lambda f: f["type"] == "ack" and f["id"] == cid)
        assert ack["payload"]["ok"] is True
        assert ack["payload"]["result"] == {"text": "長い発話"}
        assert asr.received == [wav]


async def test_asr_transcribe_rejects_oversized_audio(tmp_path):
    """Audio past MAX_AUDIO_BYTES is refused at the command layer — it must
    never reach the ASR engine (and the ws frame cap must be the layer above,
    not the thing doing this job)."""
    import base64
    asr = FakeAsr()
    async with Core(FakeLlm(), tmp_path, asr=asr) as core, Client(core.url) as c:
        wav = _fake_wav(payload_bytes=9_999_989)  # 12-byte header → 10_000_001 total, one over the cap
        cid = await c.command("asr.transcribe", {"audio": base64.b64encode(wav).decode()})
        err = await c.recv_until(lambda f: f["type"] == "error" and f["id"] == cid,
                                 timeout=15.0)
        assert err["payload"]["code"] == "bad_envelope"
        assert "too large" in err["payload"]["message"]
        assert asr.received == []


# --- skill.list / skill.run (F3 self-extending tools, first slice) --------------

def _skill_library(tmp_path):
    """A real SkillLibrary (real bwrap sandbox) with one echo skill."""
    import hashlib
    from ghost_runner_core.skills.library import SkillLibrary
    code = ('import json, sys\n'
            'print(json.dumps({"echo": json.load(sys.stdin)}, ensure_ascii=False))\n')
    d = tmp_path / "skills" / "echo"
    d.mkdir(parents=True)
    (d / "skill.py").write_text(code, encoding="utf-8")
    (d / "manifest.toml").write_text(
        'name = "echo"\nversion = "0.1.0"\ndescription = "echoes args"\n'
        f'timeout_s = 5.0\nsha256 = "{hashlib.sha256(code.encode()).hexdigest()}"\n'
        'provenance = "test"\n', encoding="utf-8")
    lib = SkillLibrary(tmp_path / "skills")
    lib.load()
    return lib


async def test_skill_list_and_run_roundtrip(tmp_path):
    """A skill invocation goes client → ws → orchestrator → bwrap sandbox → back."""
    skills = _skill_library(tmp_path)
    async with Core(FakeLlm(), tmp_path, skills=skills) as core, Client(core.url) as c:
        cid = await c.command("skill.list", {})
        ack = await c.recv_until(lambda f: f["type"] == "ack" and f["id"] == cid)
        assert ack["payload"]["ok"] is True
        assert ack["payload"]["result"]["skills"] == [
            {"name": "echo", "version": "0.1.0", "description": "echoes args"}]

        cid = await c.command("skill.run", {"name": "echo", "args": {"挨拶": "こんにちは"}})
        ack = await c.recv_until(lambda f: f["type"] == "ack" and f["id"] == cid,
                                 timeout=15.0)
        assert ack["payload"]["ok"] is True
        assert ack["payload"]["result"] == {"result": {"echo": {"挨拶": "こんにちは"}}}


async def test_skill_commands_unconfigured_are_unavailable(tmp_path):
    async with Core(FakeLlm(), tmp_path) as core, Client(core.url) as c:
        for name, args in (("skill.list", {}), ("skill.run", {"name": "echo"})):
            cid = await c.command(name, args)
            err = await c.recv_until(lambda f: f["type"] == "error" and f["id"] == cid)
            assert err["payload"]["code"] == "unavailable"


async def test_skill_run_rejects_bad_input(tmp_path):
    skills = _skill_library(tmp_path)
    async with Core(FakeLlm(), tmp_path, skills=skills) as core, Client(core.url) as c:
        # Unknown skill.
        cid = await c.command("skill.run", {"name": "nonexistent"})
        err = await c.recv_until(lambda f: f["type"] == "error" and f["id"] == cid)
        assert err["payload"]["code"] == "bad_envelope"
        assert "unknown skill" in err["payload"]["message"]
        # Missing name.
        cid = await c.command("skill.run", {})
        err = await c.recv_until(lambda f: f["type"] == "error" and f["id"] == cid)
        assert err["payload"]["code"] == "bad_envelope"
        # args not an object.
        cid = await c.command("skill.run", {"name": "echo", "args": "text"})
        err = await c.recv_until(lambda f: f["type"] == "error" and f["id"] == cid)
        assert err["payload"]["code"] == "bad_envelope"


async def test_skill_failure_surfaces_as_internal(tmp_path):
    """A skill that dies inside the sandbox comes back as a clean protocol
    error, never a hung command or a dropped connection."""
    skills = _skill_library(tmp_path)
    code_path = skills.get("echo").code_path
    code = 'import sys; print("skill blew up (test)", file=sys.stderr); sys.exit(1)\n'
    code_path.write_text(code, encoding="utf-8")
    import hashlib
    manifest = code_path.parent / "manifest.toml"
    manifest.write_text(manifest.read_text().replace(
        skills.get("echo").sha256, hashlib.sha256(code.encode()).hexdigest()))
    skills.load()  # reload so the new hash is the trusted one
    async with Core(FakeLlm(), tmp_path, skills=skills) as core, Client(core.url) as c:
        cid = await c.command("skill.run", {"name": "echo"})
        err = await c.recv_until(lambda f: f["type"] == "error" and f["id"] == cid,
                                 timeout=15.0)
        assert err["payload"]["code"] == "internal"
        assert "skill blew up (test)" in err["payload"]["message"]


async def test_skill_tampered_at_runtime_surfaces_as_unavailable(tmp_path):
    """The rug-pull guard at the protocol layer: skill.py edited on disk after
    the library was loaded (manifest hash left stale) must come back to the
    client as "unavailable" — an operator/integrity problem, not a skill crash."""
    skills = _skill_library(tmp_path)
    code_path = skills.get("echo").code_path
    async with Core(FakeLlm(), tmp_path, skills=skills) as core, Client(core.url) as c:
        code_path.write_text(code_path.read_text() + "# rug-pull\n", encoding="utf-8")
        cid = await c.command("skill.run", {"name": "echo"})
        err = await c.recv_until(lambda f: f["type"] == "error" and f["id"] == cid,
                                 timeout=15.0)
        assert err["payload"]["code"] == "unavailable"
        assert "changed on disk" in err["payload"]["message"]


# -- M1 voice output: audio_meta + binary TTS PCM over the real socket ---------


class FakeTts:
    """Duck-types tts.client.TtsClient for the orchestrator: echoes each
    sentence's UTF-8 bytes as two PCM chunks. Optionally hangs or fails."""

    def __init__(self, block_on: str | None = None, fail_on: str | None = None):
        self.sample_rate = 44100
        self.requests: list[str] = []
        self.block_on = block_on
        self.fail_on = fail_on

    async def synth_stream(self, text: str):
        self.requests.append(text)
        if text == self.fail_on:
            raise TtsError(f"synthesis failed on {text!r} (test)")
        if text == self.block_on:
            await asyncio.Event().wait()
        data = text.encode("utf-8")
        half = (len(data) // 4) * 2  # even split, sample-aligned
        yield data[:half]
        await asyncio.sleep(0)
        yield data[half:]


async def test_voice_turn_streams_audio_with_meta(tmp_path):
    """One chat turn = tokens AND one even-id PCM stream: audio_meta open →
    sliced binary frames → audio_meta close, all bound to the turn."""
    tts = FakeTts()
    llm = FakeLlm(deltas=("こんにちは。", "元気です。"))
    async with Core(llm, tmp_path, tts=tts) as core, Client(core.url) as c:
        assert "audio_meta" in c.hello_ack["payload"]["emits"]
        cid = await c.command("chat.send", {"text": "やあ"})
        ack = await c.recv_until(lambda f: f["type"] == "ack" and f["id"] == cid)
        turn = ack["payload"]["result"]["turn"]
        await c.recv_until(lambda f: f["type"] == "turn"
                           and f["payload"]["event"] == "done" and f["turn"] == turn)

        metas = [f for f in c.frames if f["type"] == "audio_meta"]
        assert len(metas) == 2
        opening, closing = metas
        assert opening["turn"] == turn and closing["turn"] == turn
        stream = opening["payload"]["stream"]
        assert stream % 2 == 0  # core-assigned S→C ids are even (§A7.4)
        assert opening["payload"] == {"stream": stream, "codec": "pcm_s16le",
                                      "sample_rate": 44100, "channels": 1, "last": False}
        assert closing["payload"]["last"] is True

        assert tts.requests == ["こんにちは。", "元気です。"]
        pcm = b""
        for i, (kind, stream_id, seq, payload) in enumerate(c.binary):
            assert (kind, stream_id, seq) == (KIND_TTS_PCM, stream, i)
            pcm += payload
        assert pcm == "こんにちは。".encode() + "元気です。".encode()


async def test_voice_barge_in_cancels_speech(tmp_path):
    """turn.cancel during synthesis kills the audio stream: the turn reports
    cancelled and no closing audio_meta ever arrives (§A5.2)."""
    tts = FakeTts(block_on="二文目です。")
    llm = FakeLlm(deltas=("一文目です。", "二文目です。", "続き"))
    async with Core(llm, tmp_path, tts=tts) as core, Client(core.url) as c:
        cid = await c.command("chat.send", {"text": "やあ"})
        ack = await c.recv_until(lambda f: f["type"] == "ack" and f["id"] == cid)
        turn = ack["payload"]["result"]["turn"]
        # Wait for the first sentence's audio, proving synthesis is mid-turn
        # (it may already have raced in while we waited for the ack).
        if not any(f["type"] == "audio_meta" for f in c.frames):
            await c.recv_until(lambda f: f["type"] == "audio_meta", timeout=5.0)

        await c.command("turn.cancel", {"turn": turn})
        await c.recv_until(lambda f: f["type"] == "turn"
                           and f["payload"]["event"] == "cancelled" and f["turn"] == turn)
        # Settle barrier: a command round-trip proves every frame the core sent
        # before this ack (including any stray closing meta) has been received.
        sync_id = await c.command("stream.sync", {"stream": f"t{turn}"})
        await c.recv_until(lambda f: f["type"] == "ack" and f["id"] == sync_id)
        metas = [f for f in c.frames if f["type"] == "audio_meta"]
        assert [m["payload"]["last"] for m in metas] == [False]


async def test_tts_failure_is_an_honest_turn_error(tmp_path):
    """Synthesis dying mid-turn is a loud ERROR, never a silently mute reply."""
    tts = FakeTts(fail_on="二文目です。")
    llm = FakeLlm(deltas=("一文目です。", "二文目です。", "です。"))
    async with Core(llm, tmp_path, tts=tts) as core, Client(core.url) as c:
        cid = await c.command("chat.send", {"text": "やあ"})
        await c.recv_until(lambda f: f["type"] == "ack" and f["id"] == cid)
        # ERROR state broadcasts first, then the turn's end event (event "done"
        # with reason "error") — wait for the later one so both are collected.
        await c.recv_until(lambda f: f["type"] == "turn"
                           and f["payload"]["event"] == "done"
                           and f["payload"].get("reason") == "error", timeout=5.0)
        err_states = [f for f in c.frames if f["type"] == "state"
                      and f["payload"]["state"] == "ERROR"]
        assert err_states and "TTS error" in err_states[-1]["payload"]["message"]


async def test_text_only_core_sends_no_audio(tmp_path):
    """Without [tts] the turn works exactly as before — no meta, no binary."""
    async with Core(FakeLlm(), tmp_path) as core, Client(core.url) as c:
        cid = await c.command("chat.send", {"text": "やあ"})
        ack = await c.recv_until(lambda f: f["type"] == "ack" and f["id"] == cid)
        turn = ack["payload"]["result"]["turn"]
        await c.recv_until(lambda f: f["type"] == "turn"
                           and f["payload"]["event"] == "done" and f["turn"] == turn)
        assert not [f for f in c.frames if f["type"] == "audio_meta"]
        assert not c.binary
