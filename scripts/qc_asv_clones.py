"""Do the ASVspoof-speaker clones actually match their source speaker?

Pitch correlation across speakers is the check: if XTTS cloned properly,
a speaker's genuine F0 and their clone's F0 should track together across
the 107 speakers (male speakers cloned low, female cloned high). A weak
correlation would mean XTTS is producing some default voice regardless of
reference, which is exactly the failure MLAAD had.
"""

import csv
import random

import librosa
import numpy as np

from src.data.asvspoof_protocol import load_bonafide_by_speaker
from src.data.loaders import TARGET_SR, load_audio


def median_f0(x: np.ndarray) -> float:
    v = librosa.yin(x, fmin=60, fmax=400, sr=TARGET_SR)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


if __name__ == "__main__":
    rng = random.Random(0)
    by_spk = load_bonafide_by_speaker()

    clones = {}
    with open("data/processed/asv_clones/manifest.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="|"):
            clones.setdefault(row["speaker"], []).append(row["path"])

    genuine_f0, clone_f0, names = [], [], []
    for spk in sorted(clones):
        g = [median_f0(load_audio(p)) for p in rng.sample(by_spk[spk], min(3, len(by_spk[spk])))]
        c = [median_f0(load_audio(p)) for p in clones[spk][:2]]
        g = [v for v in g if np.isfinite(v)]
        c = [v for v in c if np.isfinite(v)]
        if not g or not c:
            continue
        genuine_f0.append(np.median(g))
        clone_f0.append(np.median(c))
        names.append(spk)

    genuine_f0 = np.array(genuine_f0)
    clone_f0 = np.array(clone_f0)
    r = float(np.corrcoef(genuine_f0, clone_f0)[0, 1])
    err = np.abs(genuine_f0 - clone_f0)

    print(f"speakers checked: {len(genuine_f0)}")
    print(f"genuine F0 range: {genuine_f0.min():.0f}-{genuine_f0.max():.0f} Hz")
    print(f"clone   F0 range: {clone_f0.min():.0f}-{clone_f0.max():.0f} Hz")
    print(f"\npitch correlation genuine vs clone: r = {r:.3f}")
    print(f"absolute F0 error: median={np.median(err):.1f} Hz  mean={err.mean():.1f} Hz")
    print("(high r = XTTS tracked each speaker's voice; near 0 = default voice regardless of reference)")

    worst = np.argsort(-err)[:5]
    print("\nworst-matched speakers:")
    for i in worst:
        print(f"  {names[i]}: genuine={genuine_f0[i]:6.1f}  clone={clone_f0[i]:6.1f}")
