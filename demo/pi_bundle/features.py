"""The 56 hand-crafted features, standalone for the Raspberry Pi.

Verbatim copies of src/features/spectral.py, reverb.py, residual.py and the
two src/sim/acoustics.py helpers they use, so the Pi needs no repo checkout.
demo/finetune.py imports THIS file for training, so train-time and Pi-time
features cannot drift apart. demo/test_features_match.py checks the copy
still equals src/ on real clips.

One addition over src/: prepare_clip() normalises loudness. Live speech and
loudspeaker playback reach the mic at different levels, and mfcc0 is an
energy term, so without this "loud means fake" (or the reverse) is a free
shortcut that would not hold in a different room.
"""

import librosa
import numpy as np
from scipy.signal import hilbert, lfilter
from scipy.stats import kurtosis

SR = 16000
CLIP_SECONDS = 2.5
CLIP_SAMPLES = int(SR * CLIP_SECONDS)
TRIM_TOP_DB = 30.0
MIN_SPEECH_FRACTION = 0.6
TARGET_RMS_DB = -26.0

FEATURE_NAMES = (
    [f"mfcc{i}_{s}" for i in range(20) for s in ("mean", "std")]
    + [f"{n}_{s}" for n in ("centroid", "bandwidth", "rolloff", "flatness", "zcr") for s in ("mean", "std")]
    + ["spectral_tilt", "hf_energy", "modulation_depth"]
    + ["residual_energy_ratio", "residual_kurtosis", "residual_flatness"]
)


# ---------------------------------------------------------------- clip prep

def speech_fraction(x: np.ndarray) -> float:
    intervals = librosa.effects.split(x, top_db=TRIM_TOP_DB)
    return sum(e - s for s, e in intervals) / max(1, len(x))


def prepare_clip(x: np.ndarray) -> np.ndarray:
    """Exactly CLIP_SAMPLES long, loudness-normalised. Caller guarantees
    len(x) >= CLIP_SAMPLES."""
    x = x[:CLIP_SAMPLES].astype(np.float32)
    rms = np.sqrt(np.mean(x**2)) + 1e-9
    gain = 10 ** (TARGET_RMS_DB / 20) / rms
    return np.clip(x * gain, -1.0, 1.0)


# ---------------------------------------------------------------- spectral

def extract_spectral_features(x: np.ndarray, sr: int, n_mfcc: int = 20) -> dict[str, float]:
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    centroid = librosa.feature.spectral_centroid(y=x, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=x)[0]
    zcr = librosa.feature.zero_crossing_rate(x)[0]

    features = {}
    for i in range(n_mfcc):
        features[f"mfcc{i}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc{i}_std"] = float(np.std(mfcc[i]))
    for name, arr in [
        ("centroid", centroid),
        ("bandwidth", bandwidth),
        ("rolloff", rolloff),
        ("flatness", flatness),
        ("zcr", zcr),
    ]:
        features[f"{name}_mean"] = float(np.mean(arr))
        features[f"{name}_std"] = float(np.std(arr))
    return features


# ---------------------------------------------------------------- reverb

def spectral_tilt_db_per_khz(x: np.ndarray, sr: int, lo_hz: float = 100, hi_hz: float = 7500) -> float:
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    db = 10 * np.log10(spec[mask] ** 2 + 1e-12)
    slope, _ = np.polyfit(freqs[mask] / 1000, db, 1)
    return float(slope)


def hf_energy_ratio_db(x: np.ndarray, sr: int, cutoff_hz: float = 4000) -> float:
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    hf = np.sum(spec[freqs >= cutoff_hz])
    total = np.sum(spec)
    return float(10 * np.log10((hf + 1e-12) / (total + 1e-12)))


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


# ---------------------------------------------------------------- LP residual

LPC_ORDER = 16
FRAME_LENGTH = 400
HOP_LENGTH = 160
MAX_COEFF_MAGNITUDE = 20.0


def _fit_lpc_robust(frame: np.ndarray, order: int) -> np.ndarray:
    for try_order in [order, order // 2, order // 4, 1]:
        if try_order < 1:
            break
        a = librosa.lpc(frame, order=try_order)
        if np.all(np.isfinite(a)) and np.max(np.abs(a)) < MAX_COEFF_MAGNITUDE:
            return a
    return np.array([1.0])


def compute_lp_residual(x: np.ndarray, order: int = LPC_ORDER) -> np.ndarray:
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


# ---------------------------------------------------------------- all 56

def extract_all(x: np.ndarray, sr: int = SR) -> dict[str, float]:
    f = {}
    f.update(extract_spectral_features(x, sr))
    f.update(extract_reverb_features(x, sr))
    f.update(extract_residual_features(x))
    return f


def feature_vector(x: np.ndarray) -> np.ndarray:
    """prepare_clip + all 56 features, in the model's column order."""
    f = extract_all(prepare_clip(x))
    return np.array([f[n] for n in FEATURE_NAMES], dtype=np.float64)
