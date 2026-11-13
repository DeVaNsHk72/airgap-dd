"""Acoustic descriptors computed directly from a room impulse response:
T60, C50, spectral tilt, HF energy. This is what G3 actually validates —
comparing these four distributions between our training RIR pool and
ReplayDF's real measured rooms, not comparing full rendered audio.

Standard definitions (ISO 3382 for T60/C50), not something invented here.
"""

import numpy as np


def estimate_t60(rir: np.ndarray, sr: int) -> float:
    """T60 via Schroeder backward integration, extrapolated from the
    T30 slope (fit between -5dB and -35dB of the decay curve, then
    scaled x2 to reach -60dB). Real RIRs rarely have a clean 60dB of
    dynamic range above their noise floor, so extrapolating from a
    shallower, more reliably-measured range is the standard approach.
    """
    energy = rir[::-1] ** 2
    schroeder = np.cumsum(energy)[::-1]
    schroeder_db = 10 * np.log10(schroeder / (schroeder[0] + 1e-12) + 1e-12)

    t = np.arange(len(rir)) / sr
    mask = (schroeder_db <= -5) & (schroeder_db >= -35)
    if mask.sum() < 2:
        return float("nan")

    slope, intercept = np.polyfit(t[mask], schroeder_db[mask], 1)
    if slope >= 0:
        return float("nan")
    return float(-60 / slope)


def compute_c50(rir: np.ndarray, sr: int) -> float:
    """Clarity index: ratio (dB) of energy in the first 50ms after the
    direct-path peak to everything after. t=0 is the RIR's own peak."""
    peak_idx = int(np.argmax(np.abs(rir)))
    rir = rir[peak_idx:]
    split = int(0.050 * sr)
    early = np.sum(rir[:split] ** 2)
    late = np.sum(rir[split:] ** 2)
    return float(10 * np.log10((early + 1e-12) / (late + 1e-12)))


def spectral_tilt_db_per_khz(rir: np.ndarray, sr: int, lo_hz: float = 100, hi_hz: float = 7500) -> float:
    """Slope of the RIR's log-magnitude spectrum vs frequency, in dB/kHz.
    Negative means energy rolls off at high frequency (typical of real
    rooms: air and material absorption both increase with frequency)."""
    spec = np.abs(np.fft.rfft(rir))
    freqs = np.fft.rfftfreq(len(rir), 1 / sr)
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    db = 10 * np.log10(spec[mask] ** 2 + 1e-12)
    slope, _ = np.polyfit(freqs[mask] / 1000, db, 1)
    return float(slope)


def hf_energy_ratio_db(rir: np.ndarray, sr: int, cutoff_hz: float = 4000) -> float:
    """Fraction of total RIR energy above cutoff_hz, expressed in dB
    relative to total energy (always <= 0 dB)."""
    spec = np.abs(np.fft.rfft(rir)) ** 2
    freqs = np.fft.rfftfreq(len(rir), 1 / sr)
    hf = np.sum(spec[freqs >= cutoff_hz])
    total = np.sum(spec)
    return float(10 * np.log10((hf + 1e-12) / (total + 1e-12)))
