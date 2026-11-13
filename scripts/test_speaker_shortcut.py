"""Can a single speaker-identity feature (median F0) separate our
bona fide from our fake? If yes, the training task is solvable by speaker
identification rather than deepfake detection — a shortcut that cannot
transfer to any corpus with different speakers.

CLAUDE.md section 9 "corpus shortcut" in a subtler form: same corpora,
but systematically different VOICES between the two classes.
"""

import random

import librosa
import numpy as np

from src.data.loaders import TARGET_SR, load_audio
from src.data.protocols import build_bonafide_manifest, build_fake_manifest
from src.eval.metrics import compute_eer


def median_f0(x: np.ndarray) -> float:
    f0 = librosa.yin(x, fmin=60, fmax=400, sr=TARGET_SR)
    f0 = f0[np.isfinite(f0)]
    return float(np.median(f0)) if f0.size else float("nan")


if __name__ == "__main__":
    random.seed(0)
    bf = [u for u in build_bonafide_manifest() if u.split == "dev"]
    fk = [u for u in build_fake_manifest() if u.split == "dev"]
    bona = random.sample(bf, 150)
    fake = random.sample(fk, 150)

    bona_f0 = np.array([median_f0(load_audio(u.path)) for u in bona])
    fake_f0 = np.array([median_f0(load_audio(u.path)) for u in fake])
    bona_f0 = bona_f0[np.isfinite(bona_f0)]
    fake_f0 = fake_f0[np.isfinite(fake_f0)]

    print(f"bona fide median F0: {np.median(bona_f0):.1f} Hz (n={len(bona_f0)})")
    print(f"fake      median F0: {np.median(fake_f0):.1f} Hz (n={len(fake_f0)})")

    # EER using pitch ALONE as the score (try both directions, report best)
    eer_a, _ = compute_eer(bona_f0, fake_f0)
    eer_b, _ = compute_eer(-bona_f0, -fake_f0)
    best = min(eer_a, eer_b)
    print(f"\nEER using median F0 ALONE as the classifier: {best*100:.2f}%")
    print("(a low number here means the task is solvable by pitch/speaker identity,")
    print(" i.e. the model never has to learn anything about synthesis artifacts)")
