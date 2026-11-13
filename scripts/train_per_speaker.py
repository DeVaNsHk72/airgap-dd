"""Per-speaker detectors: the actual IoT setup.

Not a global model conditioned on enrolment (that was the previous
experiment). Here each speaker gets their OWN classifier, trained on that
speaker's genuine audio versus clones of that speaker — which is what a
device "learning its owner" would mean. Deployment story: at enrolment the
device synthesises clones of its owner and trains a personal detector.

Uses M-AILABS rather than ASVspoof deliberately. Genuine audio is book
readings and the clones speak M-AILABS book text, so content domain
matches — the confound that appears to have killed the ASVspoof result
(11.25% locally, 47.55% cross-corpus). Durations are also close
(6.34s vs 7.14s), and w2v2.py now crops to fixed length regardless.

Replay augmentation applied: the IoT threat is over-the-air.
"""

import random
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.data.clone_protocol import build_clone_fake_manifest
from src.data.protocols import build_bonafide_manifest
from src.eval.metrics import compute_eer
from src.features.extract_w2v2 import extract_w2v2_for_manifest

N_PER_CLASS_TRAIN = 700
N_PER_CLASS_TEST = 300
AUG_SEED_TRAIN = 542
AUG_SEED_TEST = 543
import os
RESAMPLE_MATCH = os.environ.get('RESAMPLE_MATCH', '0') == '1'


def run_speaker(spk, bona, clones, rir_paths):
    rng = random.Random(0)
    b = [u for u in bona if u.speaker == spk]
    c = [u for u in clones if u.speaker == spk]
    need = N_PER_CLASS_TRAIN + N_PER_CLASS_TEST
    if len(b) < need or len(c) < need:
        print(f"  {spk}: insufficient data (bona {len(b)}, clone {len(c)})")
        return None

    b = rng.sample(b, need)
    c = rng.sample(c, need)
    tr = b[:N_PER_CLASS_TRAIN] + c[:N_PER_CLASS_TRAIN]
    te = b[N_PER_CLASS_TRAIN:] + c[N_PER_CLASS_TRAIN:]

    X_tr, y_tr = extract_w2v2_for_manifest(tr, seed=AUG_SEED_TRAIN, rir_paths=rir_paths, resample_match=RESAMPLE_MATCH)
    X_te, y_te = extract_w2v2_for_manifest(te, seed=AUG_SEED_TEST, rir_paths=rir_paths, resample_match=RESAMPLE_MATCH)

    model = lgb.LGBMClassifier(n_estimators=400, num_leaves=31, random_state=0, verbose=-1)
    model.fit(X_tr, y_tr)
    s = model.predict_proba(X_te)[:, 1]
    eer, _ = compute_eer(s[y_te == 1], s[y_te == 0])
    print(f"  {spk:18s} train={len(tr)} test={len(te)}  EER={eer*100:.2f}%", flush=True)
    return eer


if __name__ == "__main__":
    rir_paths = list(Path("data/raw/rir").rglob("*.wav"))
    bona = build_bonafide_manifest()
    clones = build_clone_fake_manifest()
    speakers = sorted({u.speaker for u in clones})

    print(f"per-speaker detectors over {len(speakers)} speakers (M-AILABS, content-matched)")
    results = {}
    for spk in speakers:
        e = run_speaker(spk, bona, clones, rir_paths)
        if e is not None:
            results[spk] = e

    if results:
        vals = np.array(list(results.values())) * 100
        print(f"\nmean EER across speakers: {vals.mean():.2f}%")
        print(f"range: {vals.min():.2f}% - {vals.max():.2f}%")
        print("\ncomparison: global enrolment model got 11.25% (our data) / 47.55% (EchoFake)")
