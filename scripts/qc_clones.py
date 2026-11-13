"""Quality control on the generated clone set before training on it.

Checks for the failure modes that would silently poison training:
silent/near-empty outputs, degenerate durations, and whether pitch still
tracks the speaker at scale (not just in the 8-clone smoke test).
"""

import random

import librosa
import numpy as np

from src.data.clone_protocol import build_clone_fake_manifest
from src.data.loaders import TARGET_SR, load_audio
from src.data.protocols import build_bonafide_manifest

GENUINE_F0 = {
    "elliot_miller": 99.5,
    "judy_bieber": 195.4,
    "mary_ann": 165.0,
    "elizabeth_klett": 197.1,
}


def median_f0(x: np.ndarray) -> float:
    v = librosa.yin(x, fmin=60, fmax=400, sr=TARGET_SR)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


if __name__ == "__main__":
    rng = random.Random(0)
    clones = build_clone_fake_manifest()
    sample = rng.sample(clones, 200)

    durations, rms_vals, bad = [], [], []
    for u in sample:
        x = load_audio(u.path)
        durations.append(len(x) / TARGET_SR)
        rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
        rms_vals.append(rms)
        if len(x) < TARGET_SR * 0.5 or rms < 1e-4:
            bad.append(u.path)

    durations = np.array(durations)
    print(f"sampled {len(sample)} clones")
    print(f"duration: min={durations.min():.2f}s median={np.median(durations):.2f}s max={durations.max():.2f}s")
    print(f"rms: min={min(rms_vals):.5f} median={np.median(rms_vals):.5f}")
    print(f"degenerate (silent or <0.5s): {len(bad)}")
    for p in bad[:5]:
        print(f"  {p}")

    print("\npitch by speaker (clone vs genuine):")
    for spk, ref_f0 in GENUINE_F0.items():
        spk_clones = [u for u in clones if u.speaker == spk]
        sub = rng.sample(spk_clones, 15)
        f0s = np.array([median_f0(load_audio(u.path)) for u in sub])
        f0s = f0s[np.isfinite(f0s)]
        print(f"  {spk:18s} genuine={ref_f0:6.1f} Hz   clones median={np.median(f0s):6.1f} Hz  (n={len(f0s)})")

    # sanity: bona fide durations for comparison
    bona = rng.sample([u for u in build_bonafide_manifest() if u.split == "train"], 50)
    bd = np.array([len(load_audio(u.path)) / TARGET_SR for u in bona])
    print(f"\nbona fide duration for comparison: median={np.median(bd):.2f}s")
