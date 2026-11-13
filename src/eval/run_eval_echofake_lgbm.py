"""Evaluate a trained LightGBM model against an EchoFake split.

Same target/nontarget convention as run_eval_echofake.py (CNN version):
target = bonafide + replay_bonafide, nontarget = fake + replay_fake.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.loaders import TARGET_SR, load_audio_from_bytes
from src.eval.metrics import compute_eer
from src.eval.run_eval_echofake import TARGET_LABELS, NONTARGET_LABELS, load_echofake_split
from src.features.residual import extract_residual_features
from src.features.reverb import extract_reverb_features
from src.features.spectral import extract_spectral_features

NON_FEATURE_COLS = {"label", "path"}


def extract_echofake_features(split: str) -> pd.DataFrame:
    df = load_echofake_split(split)
    df = df[df["label"].isin(TARGET_LABELS | NONTARGET_LABELS)].reset_index(drop=True)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"extracting EchoFake {split} features"):
        x = load_audio_from_bytes(row["path"]["bytes"], source_desc=row["utt_id"])
        feat = {}
        feat.update(extract_spectral_features(x, TARGET_SR))
        feat.update(extract_reverb_features(x, TARGET_SR))
        feat.update(extract_residual_features(x))
        feat["label"] = 1 if row["label"] in TARGET_LABELS else 0
        feat["path"] = row["utt_id"]
        rows.append(feat)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", required=True, choices=["train", "dev", "closed_set_eval", "open_set_eval"])
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    df = extract_echofake_features(args.split)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    scores = model.predict_proba(df[feature_cols])[:, 1]
    labels = df["label"].values

    eer, threshold = compute_eer(scores[labels == 1], scores[labels == 0])
    print(
        f"split={args.split} n_target={(labels==1).sum()} n_nontarget={(labels==0).sum()} "
        f"EER={eer*100:.2f}%"
    )
