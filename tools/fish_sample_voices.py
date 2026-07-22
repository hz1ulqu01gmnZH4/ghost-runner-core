"""Sample voice styles from fish-speech S2 Pro, clone the keepers with Irodori.

The S2 Pro model synthesizes with a different sampled speaker per seed when no
reference is given. This tool:
  1. Generates the reference passage with N seeds via fish (:8930).
  2. Measures mean F0 per sample; keeps those in the requested pitch range.
  3. Installs keepers into the Irodori server's voices/ dir as ghost_s<seed>.
  4. Synthesizes an audition line through Irodori's cloning path per keeper.
Both the raw fish sample (s2_seed<N>.wav) and the Irodori clone
(ghost_s<seed>.wav) land in --audition-dir for A/B listening.

Usage:
  uv run --with pyworld --with "setuptools<81" --with soundfile --with numpy \
      --with httpx python tools/fish_sample_voices.py \
      --voices-dir ~/irodori-tts-server/voices --audition-dir <dir> \
      [--seeds 11 22 33 44 55 66] [--f0-min 165] [--f0-max 320]

Requires fish api_server on :8930 AND the Irodori server on :8088.
Fails loudly on any synthesis error.
"""

from __future__ import annotations

import argparse
import io
import wave
from pathlib import Path

import httpx
import numpy as np
import pyworld
import soundfile as sf

FISH = "http://127.0.0.1:8930"
IRODORI = "http://127.0.0.1:8088"

REFERENCE_TEXT = (
    "こんにちは。私はデスクトップに住んでいる、小さな相棒です。"
    "何か手伝えることがあったら、いつでも声をかけてくださいね。"
)
AUDITION_TEXT = "はじめまして!この声、どうかな?気に入ってもらえたら嬉しいな。"


def fish_synth(client: httpx.Client, text: str, seed: int) -> tuple[np.ndarray, int]:
    resp = client.post(
        f"{FISH}/v1/tts",
        json={"text": text, "format": "wav", "streaming": False, "seed": seed,
              "max_new_tokens": 1024, "chunk_length": 300},
        timeout=600,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"fish synthesis failed (seed {seed}): "
                           f"HTTP {resp.status_code} {resp.text[:300]}")
    data, rate = sf.read(io.BytesIO(resp.content), dtype="float64")
    if data.ndim > 1:
        raise RuntimeError(f"fish returned {data.shape[1]}-channel audio; expected mono")
    return data, int(rate)


def irodori_synth(client: httpx.Client, text: str, voice: str) -> tuple[np.ndarray, int]:
    resp = client.post(
        f"{IRODORI}/v1/audio/speech",
        json={"model": "irodori-tts", "input": text, "voice": voice,
              "response_format": "wav"},
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"irodori synthesis failed (voice {voice!r}): "
                           f"HTTP {resp.status_code} {resp.text[:300]}")
    with wave.open(io.BytesIO(resp.content), "rb") as w:
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float64) / 32768.0, rate


def mean_f0(samples: np.ndarray, rate: int) -> float:
    f0, _ = pyworld.harvest(samples, rate)
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        raise RuntimeError("no voiced frames found — sample is not speech")
    return float(voiced.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voices-dir", required=True, type=Path)
    parser.add_argument("--audition-dir", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[11, 22, 33, 44, 55, 66])
    parser.add_argument("--f0-min", type=float, default=165.0)
    parser.add_argument("--f0-max", type=float, default=320.0)
    args = parser.parse_args()

    voices_dir = args.voices_dir.expanduser()
    audition_dir = args.audition_dir.expanduser()
    if not voices_dir.is_dir():
        raise SystemExit(f"voices dir does not exist: {voices_dir}")
    audition_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client() as client:
        for name, url in [("fish", f"{FISH}/v1/health"), ("irodori", f"{IRODORI}/health")]:
            health = client.get(url, timeout=10)
            if health.status_code != 200:
                raise SystemExit(f"{name} server not ready: HTTP {health.status_code}")

        keepers: list[tuple[int, float]] = []
        for seed in args.seeds:
            print(f"fish seed {seed}: synthesizing...")
            samples, rate = fish_synth(client, REFERENCE_TEXT, seed)
            f0 = mean_f0(samples, rate)
            keep = args.f0_min <= f0 <= args.f0_max
            print(f"  {samples.size / rate:.1f}s  mean F0 {f0:.0f} Hz  "
                  f"{'KEEP' if keep else 'skip (outside range)'}")
            sf.write(str(audition_dir / f"s2_seed{seed}.wav"), samples, rate,
                     subtype="PCM_16")
            if keep:
                sf.write(str(voices_dir / f"ghost_s{seed}.wav"), samples, rate,
                         subtype="PCM_16")
                keepers.append((seed, f0))

        if not keepers:
            raise SystemExit(
                "no samples fell inside the F0 range — rerun with different "
                "--seeds or a wider --f0-min/--f0-max")

        for seed, f0 in keepers:
            voice_id = f"ghost_s{seed}"
            print(f"auditioning {voice_id} through irodori cloning...")
            cloned, cloned_rate = irodori_synth(client, AUDITION_TEXT, voice_id)
            clone_f0 = mean_f0(cloned, cloned_rate)
            sf.write(str(audition_dir / f"{voice_id}.wav"), cloned, cloned_rate,
                     subtype="PCM_16")
            print(f"  fish F0 {f0:.0f} Hz -> clone F0 {clone_f0:.0f} Hz")

    print("\nlisten in", audition_dir)
    print("adopt with [tts] voice=\"ghost_s<seed>\" in config.toml")


if __name__ == "__main__":
    main()
