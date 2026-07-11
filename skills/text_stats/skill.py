"""text_stats: character statistics for Japanese-first text.

stdin:  {"text": "..."}
stdout: {"chars", "chars_no_space", "lines", "hiragana", "katakana", "kanji"}
"""
import json
import sys


def script_of(ch: str) -> str | None:
    o = ord(ch)
    if 0x3041 <= o <= 0x309F:
        return "hiragana"
    if 0x30A0 <= o <= 0x30FF:
        return "katakana"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
        return "kanji"
    return None


def main() -> None:
    args = json.load(sys.stdin)
    text = args.get("text")
    if not isinstance(text, str):
        raise ValueError("args.text must be a string")
    counts = {"hiragana": 0, "katakana": 0, "kanji": 0}
    for ch in text:
        script = script_of(ch)
        if script is not None:
            counts[script] += 1
    print(json.dumps({
        "chars": len(text),
        "chars_no_space": sum(1 for c in text if not c.isspace()),
        "lines": len(text.splitlines()),
        **counts,
    }, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # one skill contract: any failure = stderr + exit 1
        print(f"text_stats: {exc}", file=sys.stderr)
        sys.exit(1)
