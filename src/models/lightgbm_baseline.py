"""First model, per the build order: LightGBM on hand-crafted features,
for fast feedback and a feature-importance figure before committing to
the heavier CNN.

G4 leakage check runs before feature extraction, not after — no point
training on a manifest that hasn't been checked.
"""

import random
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.protocols import build_bonafide_manifest, build_fake_manifest
from src.eval.leakage_check import assert_no_heldout_leakage
from src.eval.metrics import compute_eer
from src.features.extract import extract_features_for_manifest

N_TRAIN_PER_CLASS = 5000
N_DEV_PER_CLASS = 1000
FEATURE_SEED_TRAIN = 42
FEATURE_SEED_DEV = 43
RANDOM_SEED = 0

NON_FEATURE_COLS = {"label", "path"}


def main() -> None:
    random.seed(RANDOM_SEED)

    bf = build_bonafide_manifest()
    fk = build_fake_manifest()

    train_bf = [u for u in bf if u.split == "train"]
    train_fk = [u for u in fk if u.split == "train"]
    dev_bf = [u for u in bf if u.split == "dev"]
    dev_fk = [u for u in fk if u.split == "dev"]

    train_utts = random.sample(train_bf, N_TRAIN_PER_CLASS) + random.sample(train_fk, N_TRAIN_PER_CLASS)
    dev_utts = random.sample(dev_bf, N_DEV_PER_CLASS) + random.sample(dev_fk, N_DEV_PER_CLASS)

    print("G4 leakage check...")
    assert_no_heldout_leakage(train_utts)
    assert_no_heldout_leakage(dev_utts)
    print(f"G4 PASS: {len(train_utts)} train + {len(dev_utts)} dev utterances, no held-out paths")

    rir_paths = list(Path("data/raw/rir").rglob("*.wav"))

    print("extracting train features...")
    train_df = extract_features_for_manifest(train_utts, seed=FEATURE_SEED_TRAIN, rir_paths=rir_paths)
    print("extracting dev features...")
    dev_df = extract_features_for_manifest(dev_utts, seed=FEATURE_SEED_DEV, rir_paths=rir_paths)

    feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_dev, y_dev = dev_df[feature_cols], dev_df["label"]

    print(f"training LightGBM on {X_train.shape} features...")
    model = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, random_state=RANDOM_SEED, verbose=-1)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_dev, y_dev)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    dev_scores = model.predict_proba(X_dev)[:, 1]
    target = dev_scores[y_dev.values == 1]
    nontarget = dev_scores[y_dev.values == 0]
    eer, threshold = compute_eer(target, nontarget)
    print(f"\ndev EER (internal, augmented-domain, NOT a paper-table number): {eer*100:.2f}%")

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([feature_cols[i] for i in order][::-1], importances[order][::-1])
    ax.set_xlabel("LightGBM feature importance (split count)")
    ax.set_title(f"Top 20 features — dev EER {eer*100:.2f}% (internal, augmented-domain)")
    fig.tight_layout()
    fig.savefig("results/lightgbm_feature_importance.png", dpi=130)
    print("saved results/lightgbm_feature_importance.png")


if __name__ == "__main__":
    main()
