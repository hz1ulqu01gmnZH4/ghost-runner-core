"""Sentence chunking for streaming TTS (§A4.3 TtsPipeline, PRD §8.2).

Tokens stream in faster than speech synthesizes, so the pipeline cuts the
reply into sentences and synthesizes per sentence — first audio starts after
the first sentence, not after the whole reply. Japanese-first: the primary
terminators are 。！？ and their ASCII forms; a closing quote/bracket right
after a terminator belongs to the finished sentence (「そうですね。」 must not
leak a dangling 」 into the next one). Token boundaries are arbitrary, so a
completed sentence is held until the first character that is neither a closer
nor another terminator proves it is really over — the hold lasts one token,
not one sentence, so it costs no perceptible latency.

An LLM can also ramble without any terminator; MAX_SENTENCE_CHARS force-flushes
so speech never waits forever on a terminator that may not come (and stays
comfortably under the TTS server's own text-length limits).
"""

from __future__ import annotations

_TERMINATORS = frozenset("。！？!?…\n")
# Characters that glue onto a just-completed sentence: closing quotes/brackets
# and further terminators (えっ！？ is one sentence, not two).
_GLUE = frozenset("」』）)】＞>\"'’”") | _TERMINATORS
# A "sentence" made only of punctuation/quotes/whitespace says nothing — a TTS
# model fed a bare 。 synthesizes noise, so those fragments are dropped.
_CONTENT_FREE = _GLUE | frozenset("「『（(【＜<‘“ \t　")

MAX_SENTENCE_CHARS = 200


class SentenceChunker:
    """Accumulates streamed text deltas and yields complete sentences.

    feed() returns the sentences completed by that delta (usually none or one);
    flush() returns everything still pending at end-of-stream. Whitespace-only
    fragments are dropped — there is nothing to say.
    """

    def __init__(self) -> None:
        self._buf: list[str] = []      # sentence in progress
        self._done: str | None = None  # completed, awaiting possible closers

    def feed(self, delta: str) -> list[str]:
        sentences: list[str] = []
        for ch in delta:
            if self._done is not None:
                if ch in _GLUE:
                    self._done += ch
                    continue
                self._emit(sentences)
            self._buf.append(ch)
            if ch in _TERMINATORS or len(self._buf) >= MAX_SENTENCE_CHARS:
                self._done = "".join(self._buf)
                self._buf.clear()
        return sentences

    def flush(self) -> list[str]:
        """End-of-stream: whatever is pending, finished or not, gets spoken."""
        sentences: list[str] = []
        self._emit(sentences)
        remainder = "".join(self._buf).strip()
        self._buf.clear()
        if _speakable(remainder):
            sentences.append(remainder)
        return sentences

    def _emit(self, into: list[str]) -> None:
        if self._done is None:
            return
        sentence = self._done.strip()
        self._done = None
        if _speakable(sentence):
            into.append(sentence)


def _speakable(sentence: str) -> bool:
    return any(ch not in _CONTENT_FREE for ch in sentence)
