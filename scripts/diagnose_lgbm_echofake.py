import pickle

import numpy as np
import pandas as pd

from src.data.loaders import TARGET_SR, load_audio_from_bytes
from src.eval.run_eval_echofake import load_echofake_split
from src.features.residual import extract_residual_features
from src.features.reverb import extract_reverb_features
from src.features.spectral import extract_spectral_features

NON_FEATURE_COLS = {"label", "path"}

if __name__ == "__main__":
    with open("checkpoints/lgbm_clean_baseline.pkl", "rb") as f:
        model = pickle.load(f)

    df = load_echofake_split("closed_set_eval")

    for label in ["bonafide", "replay_bonafide", "fake", "replay_fake"]:
        sub = df[df["label"] == label].sample(100, random_state=0)
        rows = []
        for _, row in sub.iterrows():
            x = load_audio_from_bytes(row["path"]["bytes"])
            feat = {}
            feat.update(extract_spectral_features(x, TARGET_SR))
            feat.update(extract_reverb_features(x, TARGET_SR))
            feat.update(extract_residual_features(x))
            rows.append(feat)
        feat_df = pd.DataFrame(rows)
        scores = model.predict_proba(feat_df)[:, 1]
        print(f"{label:20s} n={len(scores)} mean={scores.mean():.3f} std={scores.std():.3f}")
