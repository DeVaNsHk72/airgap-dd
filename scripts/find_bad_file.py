import random
import traceback
from pathlib import Path

import numpy as np

from src.data.loaders import TARGET_SR, load_audio
from src.data.protocols import build_bonafide_manifest, build_fake_manifest
from src.sim.channel import simulate_replay_channel

# exact same selection logic as train_cnn.py
N_TRAIN_PER_CLASS = 10000
N_DEV_PER_CLASS = 2000
RUN_SEED = 0

if __name__ == "__main__":
    random.seed(RUN_SEED)
    bf = build_bonafide_manifest()
    fk = build_fake_manifest()
    train_utts = (
        random.sample([u for u in bf if u.split == "train"], N_TRAIN_PER_CLASS)
        + random.sample([u for u in fk if u.split == "train"], N_TRAIN_PER_CLASS)
    )
    dev_utts = (
        random.sample([u for u in bf if u.split == "dev"], N_DEV_PER_CLASS)
        + random.sample([u for u in fk if u.split == "dev"], N_DEV_PER_CLASS)
    )

    rir_paths = list(Path("data/raw/rir").rglob("*.wav"))

    print(f"checking {len(train_utts)} train utterances for load_audio failures first...")
    bad_load = []
    for i, u in enumerate(train_utts):
        try:
            load_audio(u.path)
        except Exception:
            bad_load.append((i, u.path, traceback.format_exc()))
        if i % 2000 == 0:
            print(f"  {i}/{len(train_utts)}")

    print(f"\nload_audio failures: {len(bad_load)}")
    for i, path, tb in bad_load[:5]:
        print(f"--- idx={i} path={path} ---")
        print(tb)
