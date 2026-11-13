"""How fast can XTTS-v2 generate clones? Determines how large a
same-speaker clone set is realistic to build."""

import os
import time
from pathlib import Path

import numpy as np

os.environ["COQUI_TOS_AGREED"] = "1"

from src.data.loaders import TARGET_SR, load_audio

if __name__ == "__main__":
    import torch
    from TTS.api import TTS

    ref_dir = Path("data/raw/train_sources/mailabs/en_US/by_book/male/elliot_miller")
    refs = sorted(
        [w for w in ref_dir.rglob("*.wav") if not w.name.startswith("._")],
        key=lambda p: -p.stat().st_size,
    )
    ref = refs[0]

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

    texts = [
        "The morning light came slowly through the window and settled on the floor.",
        "He had not expected anyone to arrive so early in the day.",
        "She turned the page carefully, listening for any sound from the hallway.",
        "It was the kind of silence that made ordinary noises seem loud.",
        "They agreed to meet again once the weather had finally cleared.",
    ]

    out_dir = Path("data/processed/clone_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    # warm up (first call includes latent extraction / graph setup)
    tts.tts_to_file(text=texts[0], speaker_wav=str(ref), language="en", file_path=str(out_dir / "warm.wav"))

    t0 = time.time()
    durations = []
    for i, text in enumerate(texts):
        p = out_dir / f"bench_{i}.wav"
        tts.tts_to_file(text=text, speaker_wav=str(ref), language="en", file_path=str(p))
        durations.append(len(load_audio(p)) / TARGET_SR)
    elapsed = time.time() - t0

    print(f"\ngenerated {len(texts)} clips in {elapsed:.1f}s = {elapsed/len(texts):.2f}s per clip")
    print(f"mean audio duration: {np.mean(durations):.2f}s")
    print(f"projected time for 5,000 clips: {5000*elapsed/len(texts)/3600:.2f} hours")
    print(f"projected time for 10,000 clips: {10000*elapsed/len(texts)/3600:.2f} hours")
