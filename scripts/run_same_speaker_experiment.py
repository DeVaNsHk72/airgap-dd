"""The same-speaker experiment.

Both classes are now the same four voices (bona fide M-AILABS vs XTTS-v2
clones of those same speakers), so the model cannot solve the task by
speaker identification — it has to learn synthesis artifacts.

Trains clean and augmented LightGBM on this task, so the augmented-vs-clean
comparison can be re-run without the voice-identity confound that affects
the MLAAD-based numbers.
"""

import pickle
from pathlib import Path

import lightgbm as lgb

from src.data.clone_protocol import build_same_speaker_pairs
from src.eval.leakage_check import assert_no_heldout_leakage
from src.eval.metrics import compute_eer
from src.features.extract import extract_features_for_manifest

NON_FEATURE_COLS = {"label", "path"}
N_TRAIN_PER_CLASS = 4500
N_DEV_PER_CLASS = 3300


def run(tag: str, seed_train, seed_dev, rir_paths) -> None:
    train_utts = build_same_speaker_pairs("train", N_TRAIN_PER_CLASS, seed=0)
    dev_utts = build_same_speaker_pairs("dev", N_DEV_PER_CLASS, seed=1)
    assert_no_heldout_leakage(train_utts)
    assert_no_heldout_leakage(dev_utts)

    n_bona_tr = sum(1 for u in train_utts if u.label == "bonafide")
    n_bona_dev = sum(1 for u in dev_utts if u.label == "bonafide")
    print(f"[{tag}] train={len(train_utts)} (bona {n_bona_tr}) dev={len(dev_utts)} (bona {n_bona_dev})")

    train_df = extract_features_for_manifest(train_utts, seed=seed_train, rir_paths=rir_paths)
    dev_df = extract_features_for_manifest(dev_utts, seed=seed_dev, rir_paths=rir_paths)

    feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    model = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, random_state=0, verbose=-1)
    model.fit(
        train_df[feature_cols],
        train_df["label"],
        eval_set=[(dev_df[feature_cols], dev_df["label"])],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    scores = model.predict_proba(dev_df[feature_cols])[:, 1]
    y = dev_df["label"].values
    eer, _ = compute_eer(scores[y == 1], scores[y == 0])
    print(f"[{tag}] same-speaker dev EER: {eer*100:.2f}%")

    out = Path(f"checkpoints/lgbm_same_speaker_{tag}.pkl")
    with open(out, "wb") as f:
        pickle.dump(model, f)
    print(f"[{tag}] saved {out}")


if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    rir_paths = list(Path("data/raw/rir").rglob("*.wav"))

    run("clean", seed_train=None, seed_dev=None, rir_paths=None)
    run("augmented", seed_train=142, seed_dev=143, rir_paths=rir_paths)
