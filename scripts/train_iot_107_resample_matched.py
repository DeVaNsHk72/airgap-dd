"""Control for the resampling artifact (corrected).

An earlier version of this control monkeypatched load_audio in the parent
process. Windows spawns worker processes, which re-import modules fresh,
so the patch never reached them and the "control" silently recomputed the
identical features — it returned exactly 5.86%, giving it away. The
resampling now travels to workers as a parameter instead.

XTTS-v2 emits 24 kHz which we downsample to 16 kHz; ASVspoof genuine audio
is natively 16 kHz, leaving a ~3 dB difference in the 7-8 kHz band
(top-band energy alone separates the classes at 37.60% EER). This routes
genuine audio through the same 16k -> 24k -> 16k path so both classes
carry identical resampling history.
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


def featurise(enrol, scored, aug_seed, rir_paths):
    flat = [u for items in enrol.values() for u in items]
    E, _ = extract_w2v2_for_manifest(flat, seed=None, resample_match=True)
    cents, idx = {}, 0
    for spk, items in enrol.items():
        cents[spk] = enrollment_centroid(E[idx : idx + len(items)])
        idx += len(items)
    X, y = extract_w2v2_for_manifest(
        scored, seed=aug_seed, rir_paths=rir_paths, resample_match=True
    )
    return build_matrix(X, np.stack([cents[u.speaker] for u in scored])), y


if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    rir_paths = list(Path("data/raw/rir").rglob("*.wav"))

    enrol_tr, scored_tr = build_enrolled_split("train", N_SCORED_TRAIN, N_ENROL, seed=0)
    enrol_dev, scored_dev = build_enrolled_split("dev", N_SCORED_DEV, N_ENROL, seed=1)
    print(f"train {len(enrol_tr)} spk / dev {len(enrol_dev)} spk", flush=True)

    X_tr, y_tr = featurise(enrol_tr, scored_tr, 442, rir_paths)
    X_dev, y_dev = featurise(enrol_dev, scored_dev, 443, rir_paths)

    model = lgb.LGBMClassifier(n_estimators=600, num_leaves=63, random_state=0, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_dev, y_dev)], callbacks=[lgb.early_stopping(50, verbose=False)])
    s = model.predict_proba(X_dev)[:, 1]
    eer, _ = compute_eer(s[y_dev == 1], s[y_dev == 0])

    print(f"\nresample-matched EER (unseen speakers): {eer*100:.2f}%")
    print("uncontrolled: 5.86%   top-band-only shortcut: 37.60%")
    print("identical to 5.86% would again mean the control did not fire")

    with open("checkpoints/iot_enrolled_107_matched.pkl", "wb") as f:
        pickle.dump(model, f)
