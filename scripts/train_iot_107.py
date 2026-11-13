"""IoT enrolled-speaker detector trained over 107 speakers.

The one variable every previous result pointed at. Four different methods
all stalled at 44-51% EER with four bona fide speakers; this holds the
method fixed and raises speaker count to 107, with train/dev split by
speaker so dev users are never seen in training.

Replay augmentation is applied because the IoT threat model is inherently
over-the-air.
"""

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.data.asv_clone_protocol import build_enrolled_split
from src.eval.metrics import compute_eer
from src.features.enrollment import build_matrix, enrollment_centroid
from src.features.extract_w2v2 import extract_w2v2_for_manifest

N_ENROL = 5
N_SCORED_TRAIN = 4000
N_SCORED_DEV = 2000
AUG_SEED_TRAIN = 442
AUG_SEED_DEV = 443


def featurise(enrol, scored, aug_seed, rir_paths):
    enrol_flat = [u for items in enrol.values() for u in items]
    E, _ = extract_w2v2_for_manifest(enrol_flat, seed=None)

    centroids, idx = {}, 0
    for spk, items in enrol.items():
        centroids[spk] = enrollment_centroid(E[idx : idx + len(items)])
        idx += len(items)

    X_test, y = extract_w2v2_for_manifest(scored, seed=aug_seed, rir_paths=rir_paths)
    enrol_rows = np.stack([centroids[u.speaker] for u in scored])
    return build_matrix(X_test, enrol_rows), y


if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    rir_paths = list(Path("data/raw/rir").rglob("*.wav"))

    enrol_tr, scored_tr = build_enrolled_split("train", N_SCORED_TRAIN, N_ENROL, seed=0)
    enrol_dev, scored_dev = build_enrolled_split("dev", N_SCORED_DEV, N_ENROL, seed=1)

    print(f"train: {len(enrol_tr)} speakers, {len(scored_tr)} scored", flush=True)
    print(f"dev:   {len(enrol_dev)} speakers, {len(scored_dev)} scored", flush=True)
    assert not (set(enrol_tr) & set(enrol_dev)), "train/dev speakers overlap"
    print("speaker disjointness: OK", flush=True)

    X_tr, y_tr = featurise(enrol_tr, scored_tr, AUG_SEED_TRAIN, rir_paths)
    X_dev, y_dev = featurise(enrol_dev, scored_dev, AUG_SEED_DEV, rir_paths)
    print(f"features: train={X_tr.shape} dev={X_dev.shape}", flush=True)
    print(f"balance train={np.bincount(y_tr)} dev={np.bincount(y_dev)}", flush=True)

    model = lgb.LGBMClassifier(n_estimators=600, num_leaves=63, random_state=0, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_dev, y_dev)], callbacks=[lgb.early_stopping(50, verbose=False)])

    s = model.predict_proba(X_dev)[:, 1]
    eer, _ = compute_eer(s[y_dev == 1], s[y_dev == 0])
    print(f"\n107-speaker IoT enrolled EER (UNSEEN dev speakers, replay-augmented): {eer*100:.2f}%")
    print("(4-speaker version of this same setup: 17.25% in-domain / 44.54% on EchoFake)")

    with open("checkpoints/iot_enrolled_107.pkl", "wb") as f:
        pickle.dump(model, f)
    print("saved checkpoints/iot_enrolled_107.pkl")
