import pytest

from ghost_runner_core.envelope import Envelope
from ghost_runner_core.server.session import REPLAY_MAX_ENTRIES, ReplayLog, SessionManager
from ghost_runner_core.store import CoreStore


def test_turn_ids_survive_restart(tmp_path):
    db = tmp_path / "ghost.db"
    store = CoreStore(db)
    first = store.next_turn_id()
    second = store.next_turn_id()
    assert second == first + 1
    store.close()
    store2 = CoreStore(db)  # restart
    assert store2.next_turn_id() == second + 1  # never reused (§A4.4)
    store2.close()


def test_turn_lifecycle_rows(tmp_path):
    store = CoreStore(tmp_path / "ghost.db")
    tid = store.next_turn_id()
    store.record_turn(tid, "balloon", "s-1")
    store.finish_turn(tid, "cancelled", "user_cancel")
    with pytest.raises(ValueError):
        store.finish_turn(tid, "active")
    store.close()


def test_message_history_order_and_limit(tmp_path):
    store = CoreStore(tmp_path / "ghost.db")
    tid = store.next_turn_id()
    store.record_turn(tid, "balloon", "s-1")
    for i in range(5):
        store.append_message(tid, "user", f"m{i}")
    assert store.recent_messages(3) == [("user", "m2"), ("user", "m3"), ("user", "m4")]
    store.close()


def _entry(seq: int) -> Envelope:
    env = Envelope(type="state", payload={"state": "IDLE"}, seq=seq)
    env.ts = 10**13  # far future: age-pruning never fires in these tests
    return env


def test_replay_tail_basic():
    log = ReplayLog()
    for s in range(1, 6):
        log.append(_entry(s))
    assert [e.seq for e in log.tail(2)] == [3, 4, 5]
    assert log.tail(5) == []          # fully current
    assert log.tail(99) == []         # ahead of us (restarted client): nothing to replay


def test_replay_refuses_past_pruned_horizon():
    log = ReplayLog()
    total = REPLAY_MAX_ENTRIES + 99   # append more than the cap → oldest 99 get pruned
    for s in range(1, total + 1):
        log.append(_entry(s))
    oldest_kept = total - REPLAY_MAX_ENTRIES + 1
    assert log.tail(1) is None        # gap in pruned territory → honest refusal (§A5.5)
    tail = log.tail(oldest_kept - 1)
    assert tail is not None and tail[0].seq == oldest_kept and tail[-1].seq == total


async def test_broadcast_assigns_seq_only_to_durable():
    mgr = SessionManager()

    class FakeWs:
        def __init__(self):
            self.frames = []

        async def send(self, text):
            self.frames.append(text)

    ws = FakeWs()
    session = mgr.create_session(ws)
    await mgr.broadcast(Envelope(type="state", payload={"state": "THINKING"}))
    await mgr.broadcast(Envelope(type="token", payload={"delta": "x", "seq": 0}))
    await mgr.broadcast(Envelope(type="state", payload={"state": "SUCCESS"}))
    import json
    frames = [json.loads(f) for f in ws.frames]
    assert frames[0]["seq"] == 1
    assert "seq" not in frames[1]                 # tokens are unsequenced (§A7.1)
    assert frames[2]["seq"] == 2
    # snapshot tracks last NON-transient state (§A5.5)
    assert mgr.state_snapshot.payload["state"] == "THINKING"
    assert session.session_id == frames[0]["session"]
