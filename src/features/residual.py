"""LP residual: fit a per-frame linear-predictive filter (approximating
the vocal tract's resonances) and subtract its prediction, leaving mostly
the excitation source plus whatever channel artifacts don't fit the
vocal-tract model. See module docstring in this package's README /
CLAUDE.md discussion for why this matters for replay detection.

order=16 is a standard choice for 16kHz speech (enough poles to capture
the formant structure without also absorbing pitch harmonics into the
"vocal tract" model).
"""

import librosa
import numpy as np
from scipy.signal import lfilter
from scipy.stats import kurtosis

LPC_ORDER = 16
FRAME_LENGTH = 400  # 25ms at 16kHz
HOP_LENGTH = 160  # 10ms at 16kHz
MAX_COEFF_MAGNITUDE = 20.0  # well-conditioned LPC coefficients for real speech stay well under this


def _fit_lpc_robust(frame: np.ndarray, order: int) -> np.ndarray:
    """Burg's method (librosa.lpc) can produce exploding coefficients on
    narrow-band/resonant content — a well-documented "pole splitting"
    failure mode, not a rare fluke. Our simulator's resonant peaking-EQ
    bumps and RIR modes make this MORE likely on augmented audio than
    clean audio, so silently skipping bad frames would make the skip
    rate itself a spurious "was this replayed" shortcut.

    Falling back to a lower order instead is principled, not a hack: a
    frame too narrow-band for a stable order-16 fit is, by construction,
    adequately modeled by a lower order anyway.
    """
    for try_order in [order, order // 2, order // 4, 1]:
        if try_order < 1:
            break
        a = librosa.lpc(frame, order=try_order)
        if np.all(np.isfinite(a)) and np.max(np.abs(a)) < MAX_COEFF_MAGNITUDE:
            return a
    return np.array([1.0])  # last resort: residual = frame itself (order-0 "prediction")


def compute_lp_residual(x: np.ndarray, order: int = LPC_ORDER) -> np.ndarray:
    """Per-frame LPC analysis, residual reconstructed via overlap-add."""
    residual = np.zeros_like(x)
    window = np.hanning(FRAME_LENGTH)

    for start in range(0, len(x) - FRAME_LENGTH, HOP_LENGTH):
        frame = x[start : start + FRAME_LENGTH]
        if np.allclose(frame, 0):
            continue
        a = _fit_lpc_robust(frame, order)
        frame_residual = lfilter(a, [1.0], frame)
        residual[start : start + FRAME_LENGTH] += frame_residual * window

    return residual


def extract_residual_features(x: np.ndarray) -> dict[str, float]:
    residual = compute_lp_residual(x)

    orig_energy = np.sum(x**2) + 1e-12
    residual_energy = np.sum(residual**2)

    return {
        "residual_energy_ratio": float(residual_energy / orig_energy),
        "residual_kurtosis": float(kurtosis(residual)),
        "residual_flatness": float(
            np.exp(np.mean(np.log(np.abs(np.fft.rfft(residual)) + 1e-12)))
            / (np.mean(np.abs(np.fft.rfft(residual))) + 1e-12)
        ),
    }
