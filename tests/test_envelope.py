import json

import pytest

from ghost_runner_core.envelope import (
    E_BAD_ENVELOPE,
    E_UNSUPPORTED_VERSION,
    Envelope,
    ProtocolError,
    decode,
)


def test_round_trip():
    env = Envelope(type="state", payload={"state": "THINKING", "confidence": 0.6},
                   turn=42, session="s-x", seq=7)
    out = decode(env.encode())
    assert out.type == "state"
    assert out.turn == 42
    assert out.session == "s-x"
    assert out.seq == 7
    assert out.payload["state"] == "THINKING"
    assert out.ts > 0  # stamped at encode


def test_encode_preserves_japanese():
    env = Envelope(type="token", payload={"delta": "こんにちは"})
    assert "こんにちは" in env.encode()  # ensure_ascii=False: readable on the wire


def test_encode_omits_absent_optionals():
    obj = json.loads(Envelope(type="ping").encode())
    assert "id" not in obj and "turn" not in obj and "seq" not in obj and "session" not in obj


@pytest.mark.parametrize("raw", [
    "not json",
    "[]",
    '{"v":1}',
    '{"v":1,"type":"nope","payload":{}}',
    '{"v":1,"type":"state","payload":[]}',
    '{"v":1,"type":"state","payload":{},"turn":"42"}',
    '{"v":1,"type":"state","payload":{},"id":7}',
    '{"v":1,"type":"state","payload":{},"seq":"x"}',
])
def test_bad_envelopes_rejected(raw):
    with pytest.raises(ProtocolError) as exc:
        decode(raw)
    assert exc.value.code == E_BAD_ENVELOPE


def test_wrong_version_rejected():
    with pytest.raises(ProtocolError) as exc:
        decode('{"v":2,"type":"state","payload":{}}')
    assert exc.value.code == E_UNSUPPORTED_VERSION


def test_binary_rejected():
    with pytest.raises(ProtocolError):
        decode(b"\x01\x02\x03")
