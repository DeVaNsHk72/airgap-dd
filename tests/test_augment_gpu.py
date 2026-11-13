"""Rule 5: every run is seeded. Verify a sample's augmentation is fully
reproducible from (run_seed, epoch, index) alone, and that different
epochs produce different augmentation (the actual point of set_epoch)."""

from pathlib import Path

from src.data.augment_gpu import ReplaySimDataset
from src.data.protocols import build_bonafide_manifest

UTTERANCES = [u for u in build_bonafide_manifest() if u.split == "train"][:5]
RIR_PATHS = list(Path("data/raw/rir").rglob("*.wav"))[:200]


def test_same_seed_epoch_index_reproduces_exactly() -> None:
    ds1 = ReplaySimDataset(UTTERANCES, RIR_PATHS, run_seed=7)
    ds2 = ReplaySimDataset(UTTERANCES, RIR_PATHS, run_seed=7)
    x1, y1 = ds1[0]
    x2, y2 = ds2[0]
    assert y1 == y2
    assert (x1 == x2).all()


def test_different_epoch_gives_different_augmentation() -> None:
    ds = ReplaySimDataset(UTTERANCES, RIR_PATHS, run_seed=7)
    ds.set_epoch(0)
    x_epoch0, _ = ds[0]
    ds.set_epoch(1)
    x_epoch1, _ = ds[0]
    assert not (x_epoch0 == x_epoch1).all()


def test_different_seed_gives_different_augmentation() -> None:
    ds_a = ReplaySimDataset(UTTERANCES, RIR_PATHS, run_seed=7)
    ds_b = ReplaySimDataset(UTTERANCES, RIR_PATHS, run_seed=8)
    x_a, _ = ds_a[0]
    x_b, _ = ds_b[0]
    assert not (x_a == x_b).all()


if __name__ == "__main__":
    test_same_seed_epoch_index_reproduces_exactly()
    test_different_epoch_gives_different_augmentation()
    test_different_seed_gives_different_augmentation()
    print("test_augment_gpu: PASS")
