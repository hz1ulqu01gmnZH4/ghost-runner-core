"""Binary WS frame encode/decode (§A7.4).

Common 18-byte little-endian header, then a per-kind payload:

    [1B hdr_ver=1][1B kind][8B stream_id u64][4B seq u32][4B payload_len u32][payload]

Stream ids are assigned by the SENDER with a parity split so the two directions
can never collide: S→C streams (TTS audio) are even and core-assigned, C→S
streams (mic, perception) are odd and client-assigned. The binary header stays
turn-agnostic — turn binding travels in the audio_meta / turn text envelopes,
and receivers resolve it through their stream registry (§A7.4).

payload_len must equal exactly the bytes that follow the header; a mismatch,
an unknown kind, or a foreign hdr_ver is a protocol fault and the frame is
rejected — never trimmed or padded into shape.
"""

from __future__ import annotations

import struct

HDR_VER = 1
KIND_TTS_PCM = 0x01     # S→C, pcm_s16le; codec params from the stream's audio_meta
KIND_MIC_PCM = 0x02     # C→S, pcm_s16le 16 kHz mono, 20 ms frames
KIND_PERCEPTION = 0x03  # C→S, [4B meta_len][meta JSON][PNG bytes]

_KNOWN_KINDS = frozenset({KIND_TTS_PCM, KIND_MIC_PCM, KIND_PERCEPTION})
_HEADER = struct.Struct("<BBQII")
HEADER_SIZE = _HEADER.size  # 18

_U64_MAX = 2**64 - 1
_U32_MAX = 2**32 - 1


class FrameError(Exception):
    """A binary frame violated §A7.4. The frame is dropped, never repaired."""


def encode_frame(kind: int, stream_id: int, seq: int, payload: bytes) -> bytes:
    """Encode one binary frame. Range violations are caller bugs → ValueError."""
    if kind not in _KNOWN_KINDS:
        raise ValueError(f"unknown binary frame kind 0x{kind:02x}")
    if not 0 <= stream_id <= _U64_MAX:
        raise ValueError(f"stream_id {stream_id} out of u64 range")
    if not 0 <= seq <= _U32_MAX:
        raise ValueError(f"seq {seq} out of u32 range")
    if not payload:
        raise ValueError("binary frame payload must be non-empty")
    return _HEADER.pack(HDR_VER, kind, stream_id, seq, len(payload)) + payload


def decode_frame(frame: bytes) -> tuple[int, int, int, bytes]:
    """Decode one binary frame into (kind, stream_id, seq, payload).

    Raises FrameError on any §A7.4 violation — receivers drop the frame with an
    error, they never guess at a partial header or truncated payload.
    """
    if len(frame) < HEADER_SIZE:
        raise FrameError(f"frame shorter than the {HEADER_SIZE}-byte header ({len(frame)} bytes)")
    hdr_ver, kind, stream_id, seq, payload_len = _HEADER.unpack_from(frame)
    if hdr_ver != HDR_VER:
        raise FrameError(f"unknown binary header version {hdr_ver}")
    if kind not in _KNOWN_KINDS:
        raise FrameError(f"unknown binary frame kind 0x{kind:02x}")
    payload = frame[HEADER_SIZE:]
    if payload_len != len(payload):
        raise FrameError(
            f"payload_len {payload_len} does not match actual payload ({len(payload)} bytes)")
    return kind, stream_id, seq, payload
