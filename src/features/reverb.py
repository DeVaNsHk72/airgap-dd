"""Blind reverberation indicators — estimated directly from a recording,
with no access to a clean impulse response. This is what a real detector
has to work with: at inference time there's no separate RIR file, only
the (possibly replayed) audio itself.

Two families of feature:
- spectral tilt / HF energy, reusing acoustics.py's functions directly —
  they're defined generically over any signal, not just impulse responses.
- modulation depth: reverberation is well known to blur the onsets and
  offsets of syllables (echo tails fill in the gaps between words), which
  measurably reduces the strength of amplitude modulation at the
  syllabic rate (roughly 2-20 Hz). This is the same principle behind the
  Speech Transmission Index used in room acoustics to predict
  intelligibility loss from reverberation — more reverb, shallower
  modulation depth in that band.
"""

import numpy as np
from scipy.signal import hilbert

from src.sim.acoustics import hf_energy_ratio_db, spectral_tilt_db_per_khz


def modulation_depth(x: np.ndarray, sr: int, lo_hz: float = 2.0, hi_hz: float = 20.0) -> float:
    envelope = np.abs(hilbert(x))
    envelope = envelope - envelope.mean()
    spec = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(len(envelope), 1 / sr)
    band_energy = np.sum(spec[(freqs >= lo_hz) & (freqs <= hi_hz)] ** 2)
    total_energy = np.sum(spec**2)
    return float(band_energy / (total_energy + 1e-12))


def extract_reverb_features(x: np.ndarray, sr: int) -> dict[str, float]:
    return {
        "spectral_tilt": spectral_tilt_db_per_khz(x, sr),
        "hf_energy": hf_energy_ratio_db(x, sr),
        "modulation_depth": modulation_depth(x, sr),
    }
