import random
import time
from pathlib import Path

from src.data.augment_gpu import make_dataloader
from src.data.protocols import build_bonafide_manifest, build_fake_manifest

if __name__ == "__main__":
    random.seed(0)
    bf = [u for u in build_bonafide_manifest() if u.split == "train"]
    fk = [u for u in build_fake_manifest() if u.split == "train"]
    utterances = random.sample(bf, 500) + random.sample(fk, 500)
    rir_paths = list(Path("data/raw/rir").rglob("*.wav"))

    for n_workers in [24, 32]:
        dataset, loader = make_dataloader(utterances, rir_paths, run_seed=0, batch_size=32, num_workers=n_workers)
        dataset.set_epoch(0)
        t0 = time.time()
        n = 0
        for batch_x, batch_y in loader:
            n += len(batch_x)
        elapsed = time.time() - t0
        print(f"workers={n_workers}: {n} samples in {elapsed:.1f}s = {n/elapsed:.1f} samples/sec")
