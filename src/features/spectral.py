"""General spectral shape features: MFCCs and standard spectral summary
statistics. These are what speaker_response.py and codec.py most directly
manipulate — EQ coloration and bandwidth limiting both show up here.

Fixed-length vector per utterance (mean + std over time for each
descriptor), since the first model (LightGBM) needs a flat feature
vector, not a time-varying spectrogram.
"""

import librosa
import numpy as np


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
