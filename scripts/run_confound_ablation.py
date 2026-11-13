"""The speaker-identity confound ablation.

Condition A: fakes are XTTS-v2 clones of the SAME speaker as bona fide.
Condition B: fakes are XTTS-v2 output in a DIFFERENT (non-M-AILABS) voice.

Everything else is held fixed — generator, texts, books, split
assignments, set sizes, per-speaker composition, features, model class,
hyperparameters. So any difference in cross-corpus transfer is
attributable to speaker matching alone, which the earlier same-speaker
experiment could not establish (it changed TTS diversity at the same time).

Both conditions are trained clean and augmented, giving a 2x2.
"""

import pickle
from pathlib import Path

import lightgbm as lgb

from src.data.clone_protocol import build_same_speaker_pairs
from src.data.mismatched_protocol import build_mismatched_pairs
from src.eval.leakage_check import assert_no_heldout_leakage
from src.eval.metrics import compute_eer
from src.features.extract import extract_features_for_manifest

NON_FEATURE_COLS = {"label", "path"}
N_TRAIN_PER_CLASS = 4500
N_DEV_PER_CLASS = 3300


def run(condition: str, aug: bool, rir_paths) -> None:
    builder = build_same_speaker_pairs if condition == "matched" else build_mismatched_pairs
    train_utts = builder("train", N_TRAIN_PER_CLASS, seed=0)
    dev_utts = builder("dev", N_DEV_PER_CLASS, seed=1)
    assert_no_heldout_leakage(train_utts)
    assert_no_heldout_leakage(dev_utts)

    seed_tr = 242 if aug else None
    seed_dev = 243 if aug else None
    rp = rir_paths if aug else None
    tag = f"{condition}_{'augmented' if aug else 'clean'}"
    print(f"[{tag}] train={len(train_utts)} dev={len(dev_utts)}", flush=True)

    train_df = extract_features_for_manifest(train_utts, seed=seed_tr, rir_paths=rp)
    dev_df = extract_features_for_manifest(dev_utts, seed=seed_dev, rir_paths=rp)

    cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    model = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, random_state=0, verbose=-1)
    model.fit(
        train_df[cols],
        train_df["label"],
        eval_set=[(dev_df[cols], dev_df["label"])],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    scores = model.predict_proba(dev_df[cols])[:, 1]
    y = dev_df["label"].values
    eer, _ = compute_eer(scores[y == 1], scores[y == 0])
    print(f"[{tag}] in-domain dev EER: {eer*100:.2f}%", flush=True)

    out = Path(f"checkpoints/ablation_{tag}.pkl")
    with open(out, "wb") as f:
        pickle.dump(model, f)
    print(f"[{tag}] saved {out}", flush=True)


if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    rir_paths = list(Path("data/raw/rir").rglob("*.wav"))
    for condition in ["matched", "mismatched"]:
        for aug in [False, True]:
            run(condition, aug, rir_paths)
