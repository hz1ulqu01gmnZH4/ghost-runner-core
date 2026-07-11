"""SentenceChunker tests: JA-first sentence boundaries over arbitrary token
splits — the exact places LLM tokenizers love to cut."""

from ghost_runner_core.voice.chunker import MAX_SENTENCE_CHARS, SentenceChunker


def run(deltas: list[str]) -> list[str]:
    c = SentenceChunker()
    out: list[str] = []
    for d in deltas:
        out.extend(c.feed(d))
    out.extend(c.flush())
    return out


def test_single_ja_sentence_terminates_on_kuten():
    assert run(["こんにち", "は。"]) == ["こんにちは。"]


def test_multiple_sentences_split_correctly():
    assert run(["今日は晴れ。明日は", "雨です。行きますか？"]) == [
        "今日は晴れ。", "明日は雨です。", "行きますか？"]


def test_closer_in_next_delta_glues_to_finished_sentence():
    # The load-bearing case: token boundary between 。 and 」.
    assert run(["「そうですね。", "」次の文です。"]) == ["「そうですね。」", "次の文です。"]


def test_consecutive_terminators_stay_one_sentence():
    assert run(["えっ！", "？そうなの。"]) == ["えっ！？", "そうなの。"]


def test_ascii_terminators_and_ellipsis():
    assert run(["Wait!", " Really?", " Hmm…"]) == ["Wait!", "Really?", "Hmm…"]


def test_newline_terminates_a_sentence():
    assert run(["一行目\n二行目"]) == ["一行目", "二行目"]


def test_unterminated_remainder_flushes_at_end():
    assert run(["終わらない文"]) == ["終わらない文"]


def test_whitespace_only_fragments_are_dropped():
    assert run(["  \n", "。", "   "]) == []


def test_pending_sentence_flushes_at_end_of_stream():
    c = SentenceChunker()
    assert c.feed("はい。") == []  # held: a closer could still arrive
    assert c.flush() == ["はい。"]


def test_terminator_free_stream_force_flushes_at_max():
    long = "あ" * (MAX_SENTENCE_CHARS * 2 + 10)
    out = run([long])
    assert len(out) == 3
    assert out[0] == "あ" * MAX_SENTENCE_CHARS
    assert out[1] == "あ" * MAX_SENTENCE_CHARS
    assert out[2] == "あ" * 10
    assert "".join(out) == long


def test_one_char_deltas_reassemble():
    # Designed behavior: a terminator inside 「…」 splits — a pause there is
    # natural prosody, and quote-depth tracking would wedge on the unbalanced
    # quotes LLMs occasionally emit. The 」 still glues to its sentence.
    text = "「猫だ！」と叫んだ。すごい。"
    assert run(list(text)) == ["「猫だ！」", "と叫んだ。", "すごい。"]


def test_punctuation_only_fragment_is_dropped():
    assert run(["（…）。"]) == []
    assert run(["「にゃー。」"]) == ["「にゃー。」"]  # real content still speaks


def test_chunker_is_reusable_after_flush():
    c = SentenceChunker()
    c.feed("一。")
    assert c.flush() == ["一。"]
    assert c.feed("二。") == []
    assert c.flush() == ["二。"]
