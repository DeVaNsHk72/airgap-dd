import random

from src.data.protocols import build_bonafide_manifest, build_fake_manifest
from src.eval.leakage_check import assert_no_heldout_leakage
from src.features.extract import extract_features_for_manifest

if __name__ == "__main__":
    random.seed(0)
    bf = build_bonafide_manifest()
    fk = build_fake_manifest()
    train_utts = random.sample([u for u in bf if u.split == "train"], 5000) + random.sample(
        [u for u in fk if u.split == "train"], 5000
    )
    dev_utts = random.sample([u for u in bf if u.split == "dev"], 1000) + random.sample(
        [u for u in fk if u.split == "dev"], 1000
    )
    assert_no_heldout_leakage(train_utts)
    assert_no_heldout_leakage(dev_utts)
    print(f"G4 PASS: {len(train_utts)} train + {len(dev_utts)} dev")

    print("extracting CLEAN (seed=None) train features...")
    train_df = extract_features_for_manifest(train_utts, seed=None)
    print("extracting CLEAN dev features...")
    dev_df = extract_features_for_manifest(dev_utts, seed=None)
    print(f"train shape: {train_df.shape}  dev shape: {dev_df.shape}")
