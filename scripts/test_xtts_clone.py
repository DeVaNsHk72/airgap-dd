"""Smoke test: can XTTS-v2 clone an M-AILABS speaker's voice?

Decisive check is pitch: elliot_miller is ~99.5 Hz (male). If the clone
lands near that, cloning worked and we can build same-speaker
genuine/fake pairs. If it lands ~200 Hz, it's synthesizing in a default
voice like MLAAD's did, and this approach won't work either.
"""

import os
from pathlib import Path

import librosa
import numpy as np

os.environ["COQUI_TOS_AGREED"] = "1"

from src.data.loaders import TARGET_SR, load_audio

OUT_DIR = Path("data/processed/clone_test")


def median_f0(x: np.ndarray, sr: int = TARGET_SR) -> float:
    f0 = librosa.yin(x, fmin=60, fmax=400, sr=sr)
    f0 = f0[np.isfinite(f0)]
    return float(np.median(f0)) if f0.size else float("nan")


if __name__ == "__main__":
    from TTS.api import TTS

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # reference clip: elliot_miller (male, ~99.5 Hz)
    ref_dir = Path("data/raw/train_sources/mailabs/en_US/by_book/male/elliot_miller")
    refs = [w for w in ref_dir.rglob("*.wav") if not w.name.startswith("._")]
    # XTTS wants a reasonably long reference; pick a longer file
    refs = sorted(refs, key=lambda p: -p.stat().st_size)[:1]
    ref = refs[0]
    print(f"reference: {ref.name}")
    print(f"reference median F0: {median_f0(load_audio(ref)):.1f} Hz")

    print("\nloading XTTS-v2 (first run downloads the model)...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

    text = "The quick brown fox jumps over the lazy dog, and the story continues well into the evening."
    out_path = OUT_DIR / "clone_elliot_miller.wav"
    tts.tts_to_file(text=text, speaker_wav=str(ref), language="en", file_path=str(out_path))

    cloned = load_audio(out_path)
    print(f"\ncloned output median F0: {median_f0(cloned):.1f} Hz")
    print(f"saved: {out_path}")
