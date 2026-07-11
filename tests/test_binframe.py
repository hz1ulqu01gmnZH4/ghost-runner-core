"""Binary frame codec tests (§A7.4): encode/decode round-trip and the
fail-loud rejections — a violated header is dropped, never repaired."""

import struct

import pytest

from ghost_runner_core.binframe import (
    HDR_VER,
    HEADER_SIZE,
    KIND_MIC_PCM,
    KIND_TTS_PCM,
    FrameError,
    decode_frame,
    encode_frame,
)


def test_round_trip_tts_pcm():
    payload = b"\x01\x02\x03\x04"
    frame = encode_frame(KIND_TTS_PCM, 42, 7, payload)
    assert len(frame) == HEADER_SIZE + 4
    kind, stream_id, seq, out = decode_frame(frame)
    assert (kind, stream_id, seq, out) == (KIND_TTS_PCM, 42, 7, payload)


def test_header_layout_is_little_endian_and_18_bytes():
    frame = encode_frame(KIND_TTS_PCM, 2, 0, b"\x00\x00")
    assert HEADER_SIZE == 18
    hdr_ver, kind, stream_id, seq, payload_len = struct.unpack_from("<BBQII", frame)
    assert (hdr_ver, kind, stream_id, seq, payload_len) == (HDR_VER, 0x01, 2, 0, 2)


def test_u64_stream_id_round_trip():
    big = 2**64 - 2
    _, stream_id, _, _ = decode_frame(encode_frame(KIND_MIC_PCM, big, 1, b"x"))
    assert stream_id == big


@pytest.mark.parametrize("kind", [0x00, 0x04, 0xFF])
def test_encode_unknown_kind_is_a_caller_bug(kind):
    with pytest.raises(ValueError, match="unknown binary frame kind"):
        encode_frame(kind, 2, 0, b"x")


def test_encode_rejects_out_of_range_ids():
    with pytest.raises(ValueError, match="u64"):
        encode_frame(KIND_TTS_PCM, 2**64, 0, b"x")
    with pytest.raises(ValueError, match="u32"):
        encode_frame(KIND_TTS_PCM, 2, 2**32, b"x")
    with pytest.raises(ValueError, match="non-empty"):
        encode_frame(KIND_TTS_PCM, 2, 0, b"")


def test_decode_rejects_short_frame():
    with pytest.raises(FrameError, match="shorter than the 18-byte header"):
        decode_frame(b"\x01\x01short")


def test_decode_rejects_foreign_header_version():
    frame = bytearray(encode_frame(KIND_TTS_PCM, 2, 0, b"xx"))
    frame[0] = 9
    with pytest.raises(FrameError, match="header version 9"):
        decode_frame(bytes(frame))


def test_decode_rejects_unknown_kind():
    frame = bytearray(encode_frame(KIND_TTS_PCM, 2, 0, b"xx"))
    frame[1] = 0x7F
    with pytest.raises(FrameError, match="kind 0x7f"):
        decode_frame(bytes(frame))


def test_decode_rejects_payload_len_mismatch():
    frame = encode_frame(KIND_TTS_PCM, 2, 0, b"xxxx")
    with pytest.raises(FrameError, match="does not match"):
        decode_frame(frame[:-1])  # truncated payload
    with pytest.raises(FrameError, match="does not match"):
        decode_frame(frame + b"y")  # trailing garbage
