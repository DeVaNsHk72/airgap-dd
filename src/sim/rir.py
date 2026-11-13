"""Room impulse response convolution: the stage that actually produces
echo and reverb, by smearing the signal in time rather than reshaping or
distorting it in place.

Training-time RIR pool must never include ReplayDF's or EchoFake's own
RIRs — those are reserved for G3's distributional validation (comparing
our simulated acoustics against theirs), not for generating training
data. Using a held-out test set's actual measured rooms to build training
data would be training on the test set's acoustic transfer functions,
even though no bona fide/fake speech is directly touched. Callers must
assemble rir_paths from data/raw/rir/{openslr28,mit_ir_survey,homula}
only.
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve


def load_rir(path: Path, target_sr: int) -> np.ndarray:
    """Load a RIR, resample to target_sr, force mono, and trim leading
    samples so the direct-path peak sits at index 0.

    Without the trim, convolving introduces a pure playback delay (the
    speaker-to-mic travel time from the original measurement) that has
    nothing to do with room acoustics — it would just shift the output
    in time relative to the input.
    """
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)

    peak_idx = int(np.argmax(np.abs(x)))
    x = x[peak_idx:]

    energy = np.sqrt(np.sum(x**2))
    return (x / (energy + 1e-9)).astype(np.float32)


def apply_rir(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve x with an already-loaded, already-aligned RIR."""
    y = fftconvolve(x, rir, mode="full")[: len(x)]
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak * np.abs(x).max()
    return y.astype(np.float32)


def random_room_response(
    x: np.ndarray, sr: int, rng: np.random.Generator, rir_paths: list[Path]
) -> np.ndarray:
    """Pick one RIR at random from rir_paths and apply it."""
    path = rir_paths[rng.integers(len(rir_paths))]
    rir = load_rir(path, sr)
    return apply_rir(x, rir)
