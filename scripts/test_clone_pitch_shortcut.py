"""Did same-speaker cloning just swap one shortcut for another?

QC showed clones sit slightly higher in pitch than genuine for the female
speakers (+8 to +26 Hz). If median F0 alone separates genuine from clone,
the model can cheat on pitch instead of learning synthesis artifacts.

Comparison point: on the MLAAD-based data, pitch alone gave 43.33% EER
(barely better than chance).
"""

import random

import librosa
import numpy as np

from src.data.clone_protocol import build_clone_fake_manifest
from src.data.loaders import TARGET_SR, load_audio
from src.data.protocols import build_bonafide_manifest
from src.eval.metrics import compute_eer


def median_f0(x: np.ndarray) -> float:
    v = librosa.yin(x, fmin=60, fmax=400, sr=TARGET_SR)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


if __name__ == "__main__":
    rng = random.Random(0)
    clones = [u for u in build_clone_fake_manifest() if u.split == "dev"]
    bona = [u for u in build_bonafide_manifest() if u.split == "dev"]

    # match speaker distribution so the test isn't confounded by which
    # speakers happen to be over-represented in either class
    from collections import Counter

    clone_counts = Counter(u.speaker for u in clones)
    sel_clones, sel_bona = [], []
    for spk, _ in clone_counts.items():
        c = [u for u in clones if u.speaker == spk]
        b = [u for u in bona if u.speaker == spk]
        n = min(40, len(c), len(b))
        sel_clones += rng.sample(c, n)
        sel_bona += rng.sample(b, n)

    cf = np.array([median_f0(load_audio(u.path)) for u in sel_clones])
    bf = np.array([median_f0(load_audio(u.path)) for u in sel_bona])
    cf, bf = cf[np.isfinite(cf)], bf[np.isfinite(bf)]

    print(f"bona fide median F0: {np.median(bf):.1f} Hz (n={len(bf)})")
    print(f"clone     median F0: {np.median(cf):.1f} Hz (n={len(cf)})")

    eer_a, _ = compute_eer(bf, cf)
    eer_b, _ = compute_eer(-bf, -cf)
    print(f"\nEER using median F0 ALONE: {min(eer_a, eer_b)*100:.2f}%")
    print("(MLAAD-based data gave 43.33%. Near 50% = no pitch shortcut;")
    print(" much lower = we traded one shortcut for another.)")
