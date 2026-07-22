"""Design candidate voices for the avatar and stage them for audition.

Pipeline:
  1. Synthesize a phonetically rich JA reference passage with Irodori's
     default (no-ref) voice.
  2. WORLD-vocoder decomposition (pyworld): shift F0 and warp the spectral
     envelope (formants) independently to sculpt named candidates.
  3. Write each candidate reference into the Irodori server's voices/ dir
     (picked up per-request, no server restart).
  4. Synthesize an audition line WITH each candidate voice (exercising the
     actual cloning path) into --audition-dir for a human to pick by ear.

Usage:
  uv run --with pyworld --with soundfile --with numpy --with httpx \
      python tools/design_voice.py \
      --voices-dir ~/irodori-tts-server/voices --audition-dir <dir>

Requires the Irodori server on :8088. Fails loudly on any synthesis or
analysis error — no silent fallbacks.
"""

from __future__ import annotations

import argparse
import io
import sys
import wave
from pathlib import Path

import httpx
import numpy as np
import pyworld
import soundfile as sf

SERVER = "http://127.0.0.1:8088"

# Rich phonetic coverage, natural register, ~15 s. The reference defines the
# cloned voice's timbre AND speaking style, so it is calm, friendly prose.
REFERENCE_TEXT = (
    "こんにちは。私はデスクトップに住んでいる、小さな相棒です。"
    "今日は良い天気ですね。窓の外では、風が静かに木々を揺らしています。"
    "何か手伝えることがあったら、いつでも声をかけてください。"
    "難しい調べものも、ちょっとした雑談も、どちらも大歓迎です。"
)

AUDITION_TEXT = "はじめまして!この声、どうかな?気に入ってもらえたら嬉しいな。"

# (voice_id, F0 ratio, formant warp ratio, description)
# The Irodori default speaker is male-leaning; reaching a female register
# needs F0 x1.5-1.8 with ~+20% formants, not semitone nudges.
CANDIDATES = [
    ("ghost_e", 1.45, 1.18, "androgynous-to-female (F0 x1.45, +18% formants)"),
    ("ghost_f", 1.60, 1.22, "natural young female (F0 x1.60, +22% formants)"),
    ("ghost_g", 1.75, 1.26, "brighter girl (F0 x1.75, +26% formants)"),
]


def synth(client: httpx.Client, text: str, voice: str) -> tuple[np.ndarray, int]:
    """Synthesize via the Irodori server; returns (float64 mono samples, rate)."""
    resp = client.post(
        f"{SERVER}/v1/audio/speech",
        json={"model": "irodori-tts", "input": text, "voice": voice,
              "response_format": "wav"},
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"synthesis failed for voice={voice!r}: HTTP {resp.status_code} {resp.text[:300]}")
    with wave.open(io.BytesIO(resp.content), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise RuntimeError(
                f"unexpected wav format: channels={w.getnchannels()} width={w.getsampwidth()}")
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float64) / 32768.0, rate


def warp_voice(samples: np.ndarray, rate: int, f0_ratio: float,
               formant_ratio: float) -> np.ndarray:
    """WORLD analysis/resynthesis with independent F0 and formant scaling."""
    if f0_ratio == 1.0 and formant_ratio == 1.0:
        return samples
    f0, t = pyworld.harvest(samples, rate)
    sp = pyworld.cheaptrick(samples, f0, t, rate)
    ap = pyworld.d4c(samples, f0, t, rate)

    f0_mod = f0 * f0_ratio

    # Warp the spectral envelope along the frequency axis: sampling the
    # original envelope at freq/ratio raises formants by `ratio`.
    bins = sp.shape[1]
    src_idx = np.arange(bins) / formant_ratio
    lo = np.clip(np.floor(src_idx).astype(int), 0, bins - 1)
    hi = np.clip(lo + 1, 0, bins - 1)
    frac = src_idx - np.floor(src_idx)
    sp_mod = sp[:, lo] * (1.0 - frac) + sp[:, hi] * frac
    sp_mod = np.ascontiguousarray(sp_mod)

    out = pyworld.synthesize(f0_mod, sp_mod, ap, rate)
    peak = np.max(np.abs(out))
    if peak == 0:
        raise RuntimeError("WORLD resynthesis produced digital silence")
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    sf.write(str(path), samples, rate, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voices-dir", required=True, type=Path)
    parser.add_argument("--audition-dir", required=True, type=Path)
    args = parser.parse_args()

    voices_dir = args.voices_dir.expanduser()
    audition_dir = args.audition_dir.expanduser()
    if not voices_dir.is_dir():
        raise SystemExit(f"voices dir does not exist: {voices_dir}")
    audition_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client() as client:
        health = client.get(f"{SERVER}/health", timeout=10)
        if health.status_code != 200 or '"loaded":true' not in health.text.replace(" ", ""):
            raise SystemExit(f"irodori server not ready: {health.status_code} {health.text[:200]}")

        print("synthesizing reference passage (default voice)...")
        base, rate = synth(client, REFERENCE_TEXT, "none")
        print(f"  {base.size / rate:.1f}s at {rate} Hz")
        write_wav(audition_dir / "reference_base.wav", base, rate)

        for voice_id, f0_ratio, formant_ratio, desc in CANDIDATES:
            print(f"designing {voice_id}: {desc}")
            shaped = warp_voice(base, rate, f0_ratio, formant_ratio)
            write_wav(voices_dir / f"{voice_id}.wav", shaped, rate)

        for voice_id, _, _, desc in CANDIDATES:
            print(f"auditioning {voice_id} through the cloning path...")
            cloned, cloned_rate = synth(client, AUDITION_TEXT, voice_id)
            write_wav(audition_dir / f"{voice_id}.wav", cloned, cloned_rate)
            print(f"  {cloned.size / cloned_rate:.1f}s")

    print("\ncandidates staged:")
    for voice_id, _, _, desc in CANDIDATES:
        print(f"  {voice_id}: {desc}")
    print(f"\nlisten in {audition_dir}; adopt with [tts] voice=\"<id>\" in config.toml")


if __name__ == "__main__":
    main()
