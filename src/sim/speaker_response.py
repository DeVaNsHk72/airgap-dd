"""Loudspeaker frequency response: what the speaker does to the sound
before it ever reaches the room. Randomized per call, not a single fixed
curve — see module docstring discussion in CLAUDE.md section 9 on why
narrow simulator ranges cause overfitting to the simulator itself.

Three effects, cascaded: bass roll-off (small drivers can't move enough
air at low frequencies), a few resonance bumps (driver/cabinet
resonances), and high-frequency roll-off (driver bandwidth limit).
"""

import numpy as np
from scipy.signal import butter, sosfilt

# Randomization ranges are deliberately wide — real speakers vary this
# much or more across laptop/phone/Bluetooth hardware.
HIGHPASS_HZ_RANGE = (80.0, 300.0)
LOWPASS_HZ_RANGE = (3000.0, 7500.0)
N_RESONANCE_BUMPS_RANGE = (1, 4)
RESONANCE_HZ_RANGE = (200.0, 7000.0)
RESONANCE_GAIN_DB_RANGE = (-8.0, 8.0)
RESONANCE_Q_RANGE = (0.7, 3.0)


def _peaking_eq_sos(freq_hz: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    """RBJ Audio EQ Cookbook peaking filter, as second-order-sections.
    Boosts or cuts a band around freq_hz by gain_db, width set by q.
    """
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)

    b0 = 1 + alpha * a
    b1 = -2 * cos_w0
    b2 = 1 - alpha * a
    a0 = 1 + alpha / a
    a1 = -2 * cos_w0
    a2 = 1 - alpha / a

    b = np.array([b0, b1, b2]) / a0
    a_coeffs = np.array([1.0, a1 / a0, a2 / a0])
    return np.concatenate([b, a_coeffs]).reshape(1, 6)


def random_speaker_response(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Apply a randomized loudspeaker coloration to x."""
    highpass_hz = rng.uniform(*HIGHPASS_HZ_RANGE)
    lowpass_hz = rng.uniform(*LOWPASS_HZ_RANGE)

    sos_list = [
        butter(2, highpass_hz, btype="highpass", fs=sr, output="sos"),
        butter(2, lowpass_hz, btype="lowpass", fs=sr, output="sos"),
    ]

    n_bumps = rng.integers(N_RESONANCE_BUMPS_RANGE[0], N_RESONANCE_BUMPS_RANGE[1] + 1)
    for _ in range(n_bumps):
        freq_hz = rng.uniform(*RESONANCE_HZ_RANGE)
        gain_db = rng.uniform(*RESONANCE_GAIN_DB_RANGE)
        q = rng.uniform(*RESONANCE_Q_RANGE)
        sos_list.append(_peaking_eq_sos(freq_hz, gain_db, q, sr))

    y = x
    for sos in sos_list:
        y = sosfilt(sos, y)
    return y.astype(np.float32)
