"""Nonlinear distortion from a driven speaker/amplifier: soft-clipping
waveshaping that generates real harmonic content, which no linear filter
(speaker_response.py) can produce.

Randomized drive per call, same reasoning as speaker_response.py: a fixed
amount of distortion would let the model learn that specific artifact
rather than the general fact that replay can introduce harmonics.
"""

import numpy as np

# Drive of 1.0 is barely-there rounding of peaks; 6.0 is audibly gritty.
# Real playback ranges from "clean, quiet phone speaker" to "cheap
# speaker pushed too loud" — this range covers both ends generously.
DRIVE_RANGE = (1.0, 6.0)


def random_nonlinear_distortion(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Soft-clip x with a randomized drive amount.

    tanh(drive * x) / tanh(drive) keeps peak amplitude roughly at 1.0
    regardless of drive, so the effect is purely about waveshape, not
    also changing loudness (which would confound the comparison).
    """
    drive = rng.uniform(*DRIVE_RANGE)
    y = np.tanh(drive * x) / np.tanh(drive)
    return y.astype(np.float32)
