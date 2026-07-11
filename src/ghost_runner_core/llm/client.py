"""Async client for llama-server's OpenAI-compatible API.

Holds the streamed response so a caller (TurnManager cancel, scheduler
preemption) can abort mid-generation by cancelling the consuming task —
closing the stream frees the llama-server slot (§A4.4).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI


class LlmError(Exception):
    """llama-server unreachable or returned an error. Always surfaced, never retried invisibly."""


class LlamaClient:
    def __init__(self, base_url: str, model: str) -> None:
        self._model = model
        # llama-server ignores the key unless --api-key is set; the SDK requires one.
        self._client = AsyncOpenAI(base_url=base_url, api_key="not-needed", max_retries=0)

    async def check_reachable(self) -> list[str]:
        """Startup fail-fast (§A9): list model ids or raise LlmError."""
        try:
            models = await self._client.models.list()
        except openai.OpenAIError as exc:
            raise LlmError(f"llama-server unreachable at {self._client.base_url}: {exc}") from exc
        return [m.id for m in models.data]

    async def chat_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Yield content deltas. Cancelling the consuming task closes the HTTP
        stream (the preemption/barge-in mechanism)."""
        try:
            stream = await self._client.chat.completions.create(
                model=self._model, messages=messages, stream=True,
            )
        except openai.OpenAIError as exc:
            raise LlmError(f"chat completion failed to start: {exc}") from exc
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue  # llama-server SSE tail: usage-only chunk after finish (§3.3)
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except openai.OpenAIError as exc:
            raise LlmError(f"chat stream broke mid-generation: {exc}") from exc
        finally:
            await stream.close()
