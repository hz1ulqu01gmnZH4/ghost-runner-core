"""Promote a liked audio clip to an Irodori cloning reference, verify stability.

Lesson learned 2026-07-22: what you audition is Irodori's CLONE OUTPUT, which
can drift far from its reference (zero-shot sampling variance — e.g. a 207 Hz
reference produced a 369 Hz audition draw the user liked). To get the voice you
actually heard, promote the audition clip ITSELF to be the reference:

  uv run --with pyworld --with "setuptools<81" --with numpy --with httpx \
      python tools/promote_voice.py <liked.wav> <voice_id> \
      --voices-dir ~/irodori-tts-server/voices

The tool installs the clip, synthesizes a test line three times through the
cloning path, and reports mean-F0 stability so you can confirm the promoted
voice reproduces what you liked before switching config.toml to it.
"""

from __future__ import annotations

import argparse
import io
import shutil
import wave
from pathlib import Path

import httpx
import numpy as np
import pyworld

SERVER = "http://127.0.0.1:8088"
TEST_TEXT = "本の話でもしようか。最近、面白い物語を読んだんだ。"
RUNS = 3


def clone_f0(client: httpx.Client, voice: str) -> float:
    resp = client.post(
        f"{SERVER}/v1/audio/speech",
        json={"model": "irodori-tts", "input": TEST_TEXT, "voice": voice,
              "response_format": "wav"},
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"synthesis failed: HTTP {resp.status_code} {resp.text[:300]}")
    with wave.open(io.BytesIO(resp.content), "rb") as w:
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    samples = pcm.astype(np.float64) / 32768.0
    f0, _ = pyworld.harvest(samples, rate)
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        raise RuntimeError("clone output has no voiced frames")
    return float(voiced.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", type=Path, help="the audio you liked (wav)")
    parser.add_argument("voice_id", help="name for the promoted voice")
    parser.add_argument("--voices-dir", required=True, type=Path)
    args = parser.parse_args()

    clip = args.clip.expanduser()
    voices_dir = args.voices_dir.expanduser()
    if not clip.is_file():
        raise SystemExit(f"clip not found: {clip}")
    if not voices_dir.is_dir():
        raise SystemExit(f"voices dir not found: {voices_dir}")
    target = voices_dir / f"{args.voice_id}{clip.suffix.lower()}"
    if target.exists():
        raise SystemExit(f"voice {args.voice_id!r} already exists at {target}")

    # Source clip F0 — the identity we are trying to pin.
    with wave.open(str(clip), "rb") as w:
        rate = w.getframerate()
        src = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    f0, _ = pyworld.harvest(src, rate)
    src_f0 = float(f0[f0 > 0].mean())

    shutil.copyfile(clip, target)
    print(f"installed {target}  (source mean F0 {src_f0:.0f} Hz)")

    with httpx.Client() as client:
        results = [clone_f0(client, args.voice_id) for _ in range(RUNS)]
    for i, value in enumerate(results, 1):
        print(f"clone run {i}: mean F0 {value:.0f} Hz")
    spread = max(results) - min(results)
    drift = abs(float(np.mean(results)) - src_f0)
    print(f"spread {spread:.0f} Hz, drift from source {drift:.0f} Hz")
    print(f"adopt with:  [tts] voice = \"{args.voice_id}\"  in config.toml, then restart core")


if __name__ == "__main__":
    main()
