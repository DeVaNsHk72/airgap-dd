import random
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.data.loaders import TARGET_SR, load_audio
from src.data.protocols import build_bonafide_manifest, build_fake_manifest
from src.sim.channel import simulate_replay_channel


def _try_one(args):
    path, seed = args
    try:
        x = load_audio(path)
        rir_paths = list(Path("data/raw/rir").rglob("*.wav"))
        rng = np.random.default_rng(seed)
        y = simulate_replay_channel(x, TARGET_SR, rng, rir_paths)
        return None
    except Exception as e:
        return f"{path}\n{traceback.format_exc()}"


if __name__ == "__main__":
    random.seed(1)
    bf = [u for u in build_bonafide_manifest() if u.split == "train"]
    fk = [u for u in build_fake_manifest() if u.split == "train"]
    utterances = random.sample(bf, 1000) + random.sample(fk, 1000)
    tasks = [(u.path, i) for i, u in enumerate(utterances)]

    errors = []
    with ProcessPoolExecutor(max_workers=16) as pool:
        for i, result in enumerate(pool.map(_try_one, tasks, chunksize=4)):
            if result is not None:
                errors.append(result)
            if i % 200 == 0:
                print(f"{i}/{len(tasks)} done, {len(errors)} errors so far")

    print(f"\ntotal errors: {len(errors)} / {len(tasks)}")
    for e in errors[:5]:
        print("=" * 60)
        print(e)
