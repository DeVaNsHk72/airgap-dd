"""Cache features for the additional liveness experiments in the paper.

Run from the airgap-dd root:
    PYTHONPATH=. python ../../paper/scripts/liveness_features.py <RIRS_NOISES dir>

Same items, seeds and feature code as demo/finetune.py. Differences:
  * the impulse response pool is OpenSLR-28 only, with noise recordings removed
  * each loudspeaker item is rendered under several channel conditions
      S     speaker + distortion + room, simulated rooms   (all splits)
      R     speaker + distortion + room, measured rooms    (all splits)
      room  room only, simulated rooms                     (all splits)
      spk   speaker + distortion only                      (all splits)
"""
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from demo.pi_bundle import features as pi
from src.data.loaders import load_audio
from src.sim.nonlinear import random_nonlinear_distortion
from src.sim.rir import random_room_response
from src.sim.speaker_response import random_speaker_response

SEED = 0
SESSION_SPLIT = {"2026-10-15": "train", "2026-09-08": "dev", "2026-10-16": "test"}
RIR_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT = Path("../../paper/results/liveness_features.npz")

SIM_RIRS = sorted((RIR_ROOT / "simulated_rirs").rglob("*.wav"))
REAL_RIRS = sorted(p for p in (RIR_ROOT / "real_rirs_isotropic_noises").glob("*.wav") if "noise" not in p.name.lower())
CONDITIONS = ("S", "R", "room", "spk")


def fix(p: str) -> str:
    return p.replace("\\", "/")


def build_items() -> list[dict]:
    items = []
    with open("data/demo/my_voice/manifest.csv", newline="") as f:
        for r in csv.DictReader(f):
            split = SESSION_SPLIT[r["session"]]
            items.append({"path": fix(r["path"]), "split": split, "kind": "live", "label": 1, "replay": False})
            items.append({"path": fix(r["path"]), "split": split, "kind": "replay_genuine", "label": 0, "replay": True})
    with open("data/demo/fakes/manifest.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            items.append({"path": fix(r["path"]), "split": r["split"], "kind": f"replay_{r['generator']}",
                          "label": 0, "replay": True})
    return items


def render(x: np.ndarray, cond: str, rng: np.random.Generator) -> np.ndarray:
    if cond in ("S", "R", "spk"):
        x = random_speaker_response(x, pi.SR, rng)
        x = random_nonlinear_distortion(x, rng)
    if cond in ("S", "room"):
        x = random_room_response(x, pi.SR, rng, SIM_RIRS)
    elif cond == "R":
        x = random_room_response(x, pi.SR, rng, REAL_RIRS)
    return x


def work(args):
    i, it = args
    if it["kind"] in ("live", "replay_genuine"):
        x, _ = sf.read(it["path"], dtype="float32")
    else:
        x = load_audio(Path(it["path"]))
    if len(x) < pi.CLIP_SAMPLES:
        return None
    if not it["replay"]:
        return {"clean": pi.feature_vector(x)}
    return {c: pi.feature_vector(render(x, c, np.random.default_rng([SEED, i]))) for c in CONDITIONS}


def main():
    print(f"simulated RIRs {len(SIM_RIRS)}, measured RIRs {len(REAL_RIRS)}", flush=True)
    items = build_items()
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        res = list(pool.map(work, enumerate(items), chunksize=8))
    keep = [i for i, r in enumerate(res) if r is not None]
    print(f"kept {len(keep)}/{len(items)}", flush=True)
    out = {
        "kind": np.array([items[i]["kind"] for i in keep]),
        "split": np.array([items[i]["split"] for i in keep]),
        "label": np.array([items[i]["label"] for i in keep]),
        "path": np.array([items[i]["path"] for i in keep]),
    }
    nan = np.full(len(pi.FEATURE_NAMES), np.nan)
    for c in ("clean",) + CONDITIONS:
        out["X_" + c] = np.stack([res[i].get(c, nan) for i in keep])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **out)
    print("saved", OUT)


if __name__ == "__main__":
    main()
