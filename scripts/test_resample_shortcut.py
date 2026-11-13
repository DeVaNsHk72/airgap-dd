"""Is the 5.86% result driven by a resampling/format artifact?

Genuine audio is ASVspoof flac, natively 16 kHz. XTTS-v2 emits 24 kHz,
which load_audio resamples to 16 kHz. Resampling leaves a characteristic
rolloff near Nyquist, so a model could learn "was this resampled?" rather
than "is this synthetic" — an artifact that would not survive deployment.

Test: measure energy in the top frequency band (7-8 kHz, just below
Nyquist) and see how well that ALONE separates the classes. Near-chance
means no format shortcut; a low EER means the result is contaminated.
"""

import random

import numpy as np

from src.data.asv_clone_protocol import build_asv_clone_manifest
from src.data.asvspoof_protocol import build_asv_bonafide_manifest
from src.data.loaders import TARGET_SR, load_audio
from src.eval.metrics import compute_eer


def top_band_ratio(x: np.ndarray, lo: float = 7000.0) -> float:
    """Fraction of total energy above `lo` Hz, in dB."""
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / TARGET_SR)
    hi = spec[freqs >= lo].sum()
    tot = spec.sum()
    return float(10 * np.log10((hi + 1e-12) / (tot + 1e-12)))


if __name__ == "__main__":
    rng = random.Random(0)
    bona = [u for u in build_asv_bonafide_manifest() if u.split == "dev"]
    clones = [u for u in build_asv_clone_manifest() if u.split == "dev"]

    b = rng.sample(bona, 250)
    c = rng.sample(clones, 250)

    bv = np.array([top_band_ratio(load_audio(u.path)) for u in b])
    cv = np.array([top_band_ratio(load_audio(u.path)) for u in c])

    print(f"genuine  7-8kHz energy: median={np.median(bv):7.2f} dB")
    print(f"clone    7-8kHz energy: median={np.median(cv):7.2f} dB")
    print(f"separation: {abs(np.median(bv)-np.median(cv)):.2f} dB")

    a, _ = compute_eer(bv, cv)
    d, _ = compute_eer(-bv, -cv)
    eer = min(a, d)
    print(f"\nEER using top-band energy ALONE: {eer*100:.2f}%")
    print("near 50% = no format shortcut; low = the 5.86% is contaminated")
