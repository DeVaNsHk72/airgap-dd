"""W2V2 experiment: same-speaker task, clean vs simulated-replay training.

Same protocol as run_same_speaker_experiment.py, but with frozen W2V2
embeddings instead of hand-crafted features, to test whether pretrained
representations generalise cross-corpus where from-scratch features did
not.
"""

import pickle
from pathlib import Path

import lightgbm as lgb

from src.data.clone_protocol import build_same_speaker_pairs
from src.eval.leakage_check import assert_no_heldout_leakage
from src.eval.metrics import compute_eer
from src.features.extract_w2v2 import extract_w2v2_for_manifest

N_TRAIN_PER_CLASS = 4500
N_DEV_PER_CLASS = 3300


def run(tag: str, seed_train, seed_dev, rir_paths) -> None:
    train_utts = build_same_speaker_pairs("train", N_TRAIN_PER_CLASS, seed=0)
    dev_utts = build_same_speaker_pairs("dev", N_DEV_PER_CLASS, seed=1)
    assert_no_heldout_leakage(train_utts)
    assert_no_heldout_leakage(dev_utts)
    print(f"[{tag}] train={len(train_utts)} dev={len(dev_utts)}", flush=True)

    X_tr, y_tr = extract_w2v2_for_manifest(train_utts, seed=seed_train, rir_paths=rir_paths)
    X_dev, y_dev = extract_w2v2_for_manifest(dev_utts, seed=seed_dev, rir_paths=rir_paths)
    print(f"[{tag}] embeddings: train={X_tr.shape} dev={X_dev.shape}", flush=True)

    model = lgb.LGBMClassifier(n_estimators=400, num_leaves=31, random_state=0, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_dev, y_dev)], callbacks=[lgb.early_stopping(40, verbose=False)])

    scores = model.predict_proba(X_dev)[:, 1]
    eer, _ = compute_eer(scores[y_dev == 1], scores[y_dev == 0])
    print(f"[{tag}] W2V2 same-speaker dev EER: {eer*100:.2f}%", flush=True)

    out = Path(f"checkpoints/w2v2_lgbm_{tag}.pkl")
    with open(out, "wb") as f:
        pickle.dump(model, f)
    print(f"[{tag}] saved {out}", flush=True)


if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    rir_paths = list(Path("data/raw/rir").rglob("*.wav"))
    run("clean", seed_train=None, seed_dev=None, rir_paths=None)
    run("augmented", seed_train=142, seed_dev=143, rir_paths=rir_paths)
