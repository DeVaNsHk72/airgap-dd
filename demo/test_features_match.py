"""The Pi bundle's feature copy must equal src/ exactly, and its column
order must equal the base model's. Run: PYTHONPATH=. python demo/test_features_match.py
"""

import csv
import pickle
import random

import numpy as np
import soundfile as sf

from demo.pi_bundle import features as pi
from src.features.residual import extract_residual_features
from src.features.reverb import extract_reverb_features
from src.features.spectral import extract_spectral_features


def main() -> None:
    with open("checkpoints/lgbm_same_speaker_augmented.pkl", "rb") as f:
        base = pickle.load(f)
    assert list(base.feature_name_) == pi.FEATURE_NAMES, "column order differs from base model"

    with open("data/demo/my_voice/manifest.csv", newline="") as f:
        paths = [r["path"] for r in csv.DictReader(f)]
    for p in random.Random(0).sample(paths, 5):
        x, sr = sf.read(p, dtype="float32")
        ref = {}
        ref.update(extract_spectral_features(x, sr))
        ref.update(extract_reverb_features(x, sr))
        ref.update(extract_residual_features(x))
        got = pi.extract_all(x, sr)
        for k in pi.FEATURE_NAMES:
            assert np.isclose(ref[k], got[k], rtol=1e-6, atol=1e-9), f"{p}: {k} {ref[k]} != {got[k]}"
    print("PASS: Pi feature copy matches src/ on 5 clips, column order matches base model")


if __name__ == "__main__":
    main()
