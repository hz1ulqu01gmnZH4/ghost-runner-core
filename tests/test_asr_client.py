"""WhisperClient unit tests against an httpx MockTransport — verifies the
multipart request shape and that every failure mode raises AsrError loudly."""

import httpx
import pytest

from ghost_runner_core.asr.client import AsrError, WhisperClient

WAV = b"RIFF\x24\x00\x00\x00WAVE" + b"\x00" * 32


def make_client(handler) -> WhisperClient:
    return WhisperClient("http://asr.test", "ja",
                         transport=httpx.MockTransport(handler))


async def test_transcribe_posts_multipart_and_strips_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = request.read()
        return httpx.Response(200, json={"text": " 明日の天気を教えて \n"})

    client = make_client(handler)
    try:
        assert await client.transcribe(WAV) == "明日の天気を教えて"
    finally:
        await client.aclose()
    assert seen["path"] == "/inference"
    assert WAV in seen["body"]                    # the wav travels byte-exact
    assert b'name="language"' in seen["body"]     # multipart form fields present
    assert b"ja" in seen["body"]


async def test_transcribe_http_error_raises():
    client = make_client(lambda req: httpx.Response(500, text="boom"))
    try:
        with pytest.raises(AsrError, match="500"):
            await client.transcribe(WAV)
    finally:
        await client.aclose()


async def test_transcribe_missing_text_field_raises():
    client = make_client(lambda req: httpx.Response(200, json={"error": "nope"}))
    try:
        with pytest.raises(AsrError, match="no text field"):
            await client.transcribe(WAV)
    finally:
        await client.aclose()


async def test_transcribe_non_json_raises():
    client = make_client(lambda req: httpx.Response(200, text="<html>oops</html>"))
    try:
        with pytest.raises(AsrError, match="non-JSON"):
            await client.transcribe(WAV)
    finally:
        await client.aclose()


async def test_transcribe_connect_failure_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = make_client(handler)
    try:
        with pytest.raises(AsrError, match="request failed"):
            await client.transcribe(WAV)
    finally:
        await client.aclose()


async def test_check_reachable_ok_and_failing():
    ok = make_client(lambda req: httpx.Response(200, json={"status": "ok"}))
    try:
        await ok.check_reachable()  # must not raise
    finally:
        await ok.aclose()

    missing = make_client(lambda req: httpx.Response(404))
    try:
        with pytest.raises(AsrError, match="404"):
            await missing.check_reachable()
    finally:
        await missing.aclose()
