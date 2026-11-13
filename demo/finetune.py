"""Personalise the general detector for the Raspberry Pi liveness demo.

Task: score 1 = Devansh speaking live into the mic, 0 = audio coming out of
a loudspeaker (a deepfake of him, or a real recording of him played back).

Base model: checkpoints/lgbm_same_speaker_augmented.pkl (56 hand-crafted
features). Fine-tuning = LightGBM continued boosting (init_model): the base
trees are kept and new trees are added on Devansh's data.

Loudspeaker class is built with the project simulator's physical stages
(speaker response, driver distortion, room). The codec stage is left out on
purpose: in the demo nothing passes through a codec, so codec artefacts
would be a cue that exists in training and never at the Pi.

Split by recording day, never by clip:
  train 2026-10-15   dev 2026-09-08   test 2026-10-16
Test also holds YourTTS fakes, a generator absent from training.

Demo data only: none of these numbers belong in a paper table.
Run: PYTHONPATH=. python demo/finetune.py
"""

import csv
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import soundfile as sf

from demo.pi_bundle import features as pi
from src.data.loaders import load_audio
from src.eval.metrics import compute_eer
from src.sim.nonlinear import random_nonlinear_distortion
from src.sim.rir import random_room_response
from src.sim.speaker_response import random_speaker_response

SEED = 0
SESSION_SPLIT = {"2026-10-15": "train", "2026-09-08": "dev", "2026-10-16": "test"}
BASE_MODEL = Path("checkpoints/lgbm_same_speaker_augmented.pkl")
BUNDLE = Path("demo/pi_bundle")
RIR_PATHS = sorted(Path("data/raw/rir").rglob("*.wav"))


def simulate_loudspeaker(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = random_speaker_response(x, pi.SR, rng)
    y = random_nonlinear_distortion(y, rng)
    return random_room_response(y, pi.SR, rng, RIR_PATHS)


def build_items() -> list[dict]:
    items = []
    with open("data/demo/my_voice/manifest.csv", newline="") as f:
        for r in csv.DictReader(f):
            split = SESSION_SPLIT[r["session"]]
            items.append({"path": r["path"], "split": split, "kind": "live", "label": 1, "replay": False})
            items.append({"path": r["path"], "split": split, "kind": "replay_genuine", "label": 0, "replay": True})
    with open("data/demo/fakes/manifest.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            items.append({"path": r["path"], "split": r["split"], "kind": f"replay_{r['generator']}",
                          "label": 0, "replay": True})
    return items


def _featurise(args: tuple[int, dict]) -> np.ndarray | None:
    i, it = args
    if it["kind"] in ("live", "replay_genuine"):
        x, _ = sf.read(it["path"], dtype="float32")  # already 16 kHz, trimmed, 2.5 s
    else:
        x = load_audio(Path(it["path"]))  # TTS output: resample + same trim as everything else
    if it["replay"]:
        x = simulate_loudspeaker(x, np.random.default_rng([SEED, i]))
    if len(x) < pi.CLIP_SAMPLES:
        return None  # too short for a full window; never pad, padding is a length cue
    return pi.feature_vector(x)


def featurise(items: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as pool:
        vecs = list(pool.map(_featurise, enumerate(items), chunksize=8))
    keep = [i for i, v in enumerate(vecs) if v is not None]
    X = np.stack([vecs[i] for i in keep])
    y = np.array([items[i]["label"] for i in keep])
    kinds = np.array([items[i]["kind"] for i in keep])
    splits = np.array([items[i]["split"] for i in keep])
    print(f"featurised {len(keep)}/{len(items)} items ({len(items) - len(keep)} fakes too short)")
    return X, y, np.stack([kinds, splits], axis=1)


def report(name: str, scores: np.ndarray, y: np.ndarray, kinds: np.ndarray, thr: float) -> dict:
    live = scores[y == 1]
    out = {"eer_all": compute_eer(live, scores[y == 0])[0] * 100,
           "live_accept": float(np.mean(live >= thr) * 100)}
    line = f"{name:22s} EER all {out['eer_all']:5.2f}% | live accepted {out['live_accept']:5.1f}%"
    for k in sorted(set(kinds[y == 0])):
        neg = scores[kinds == k]
        out[f"eer_{k}"] = compute_eer(live, neg)[0] * 100
        out[f"reject_{k}"] = float(np.mean(neg < thr) * 100)
        line += f" | {k}: EER {out[f'eer_{k}']:5.2f}%, rejected {out[f'reject_{k}']:5.1f}%"
    print(line)
    return out


def main() -> None:
    items = build_items()
    X, y, meta = featurise(items)
    kinds, splits = meta[:, 0], meta[:, 1]
    tr, dv, te = (splits == "train"), (splits == "dev"), (splits == "test")
    for s, m in [("train", tr), ("dev", dv), ("test", te)]:
        print(f"{s}: {m.sum()} clips  " + ", ".join(f"{k}={np.sum(kinds[m] == k)}" for k in sorted(set(kinds[m]))))

    with open(BASE_MODEL, "rb") as f:
        base = pickle.load(f).booster_

    params = {"objective": "binary", "num_leaves": 31, "learning_rate": 0.05, "seed": SEED,
              "is_unbalance": True, "verbose": -1, "feature_pre_filter": False}
    dtrain = lgb.Dataset(X[tr], y[tr], feature_name=pi.FEATURE_NAMES, free_raw_data=False)
    ddev = lgb.Dataset(X[dv], y[dv], reference=dtrain)
    tuned = lgb.train(params, dtrain, num_boost_round=400, init_model=base, valid_sets=[ddev],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
    print(f"fine-tuned: base {base.num_trees()} trees + {tuned.num_trees() - base.num_trees()} new")

    results = {}
    for name, model in [("base (general)", base), ("fine-tuned (personal)", tuned)]:
        dev_scores = model.predict(X[dv])
        _, thr = compute_eer(dev_scores[y[dv] == 1], dev_scores[y[dv] == 0])
        print(f"\n[{name}] threshold from dev EER: {thr:.4f}")
        report("  dev (8 Sep)", dev_scores, y[dv], kinds[dv], thr)
        results[name] = {"threshold": float(thr),
                         "test": report("  TEST (16 Oct)", model.predict(X[te]), y[te], kinds[te], thr)}

    BUNDLE.mkdir(parents=True, exist_ok=True)
    tuned.save_model(str(BUNDLE / "model.txt"))
    config = {
        "threshold": results["fine-tuned (personal)"]["threshold"],
        "score_meaning": "P(live human). score >= threshold -> LIVE, else LOUDSPEAKER",
        "sample_rate": pi.SR, "clip_seconds": pi.CLIP_SECONDS,
        "min_speech_fraction": pi.MIN_SPEECH_FRACTION, "target_rms_db": pi.TARGET_RMS_DB,
        "feature_names": pi.FEATURE_NAMES, "base_model": str(BASE_MODEL),
        "lightgbm_version": lgb.__version__, "seed": SEED, "split": SESSION_SPLIT,
        "test_results": results,
    }
    (BUNDLE / "config.json").write_text(json.dumps(config, indent=2))
    print(f"\nsaved {BUNDLE / 'model.txt'} and {BUNDLE / 'config.json'}")


if __name__ == "__main__":
    main()
