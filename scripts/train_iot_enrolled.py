"""Train the enrolled-speaker (IoT) detector.

Trains only on our own speakers: M-AILABS genuine vs XTTS-v2 clones of
those same speakers. Replay augmentation is applied because the IoT
threat model is inherently over-the-air — an attacker cannot inject a
file into a smart speaker, they must play it into the microphone.

Enrolment clips are held strictly disjoint from scored clips, so no
utterance is ever compared against itself.
"""

import pickle
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.data.clone_protocol import build_clone_fake_manifest
from src.data.protocols import build_bonafide_manifest
from src.eval.leakage_check import assert_no_heldout_leakage
from src.eval.metrics import compute_eer
from src.features.enrollment import build_matrix, enrollment_centroid
from src.features.extract_w2v2 import extract_w2v2_for_manifest

N_ENROL_PER_SPEAKER = 10  # what a real device would plausibly hold
N_SCORED_PER_CLASS_TRAIN = 3000
N_SCORED_PER_CLASS_DEV = 2000
AUG_SEED_TRAIN = 342
AUG_SEED_DEV = 343


def build_split(split: str, n_per_class: int, seed: int):
    """Returns (enrol_utts_by_speaker, scored_utts) with no overlap."""
    rng = random.Random(seed)
    bona = [u for u in build_bonafide_manifest() if u.split == split]
    clones = [u for u in build_clone_fake_manifest() if u.split == split]

    by_spk: dict[str, list] = {}
    for u in bona:
        by_spk.setdefault(u.speaker, []).append(u)

    enrol: dict[str, list] = {}
    used: set = set()
    for spk, items in by_spk.items():
        picked = rng.sample(items, min(N_ENROL_PER_SPEAKER, len(items)))
        enrol[spk] = picked
        used.update(str(p.path) for p in picked)

    speakers = sorted(enrol)
    per_spk = max(1, n_per_class // len(speakers))

    scored = []
    for spk in speakers:
        spk_bona = [u for u in by_spk[spk] if str(u.path) not in used]
        spk_clone = [u for u in clones if u.speaker == spk]
        scored.extend(rng.sample(spk_bona, min(per_spk, len(spk_bona))))
        scored.extend(rng.sample(spk_clone, min(per_spk, len(spk_clone))))
    return enrol, scored


def featurise(enrol, scored, aug_seed, rir_paths):
    # enrolment is always clean: a device enrols its owner from genuine,
    # directly-captured audio, not over-the-air attack audio
    enrol_flat = [u for items in enrol.values() for u in items]
    E, _ = extract_w2v2_for_manifest(enrol_flat, seed=None)

    centroids: dict[str, np.ndarray] = {}
    idx = 0
    for spk, items in enrol.items():
        centroids[spk] = enrollment_centroid(E[idx : idx + len(items)])
        idx += len(items)

    X_test, y = extract_w2v2_for_manifest(scored, seed=aug_seed, rir_paths=rir_paths)
    enrol_rows = np.stack([centroids[u.speaker] for u in scored])
    X = build_matrix(X_test, enrol_rows)
    return X, y


if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    rir_paths = list(Path("data/raw/rir").rglob("*.wav"))

    enrol_tr, scored_tr = build_split("train", N_SCORED_PER_CLASS_TRAIN, seed=0)
    enrol_dev, scored_dev = build_split("dev", N_SCORED_PER_CLASS_DEV, seed=1)
    assert_no_heldout_leakage(scored_tr)
    assert_no_heldout_leakage(scored_dev)

    n_enrol = sum(len(v) for v in enrol_tr.values())
    print(f"train: {n_enrol} enrolment clips, {len(scored_tr)} scored", flush=True)
    print(f"dev:   {len(scored_dev)} scored", flush=True)

    X_tr, y_tr = featurise(enrol_tr, scored_tr, AUG_SEED_TRAIN, rir_paths)
    X_dev, y_dev = featurise(enrol_dev, scored_dev, AUG_SEED_DEV, rir_paths)
    print(f"features: train={X_tr.shape} dev={X_dev.shape}", flush=True)
    print(f"train balance: {np.bincount(y_tr)}  dev balance: {np.bincount(y_dev)}", flush=True)

    model = lgb.LGBMClassifier(n_estimators=500, num_leaves=31, random_state=0, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_dev, y_dev)], callbacks=[lgb.early_stopping(40, verbose=False)])

    scores = model.predict_proba(X_dev)[:, 1]
    eer, _ = compute_eer(scores[y_dev == 1], scores[y_dev == 0])
    print(f"\nIoT enrolled-speaker dev EER (our speakers, replay-augmented): {eer*100:.2f}%")

    with open("checkpoints/iot_enrolled.pkl", "wb") as f:
        pickle.dump(model, f)
    print("saved checkpoints/iot_enrolled.pkl")
