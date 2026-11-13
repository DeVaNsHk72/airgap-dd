"""Evaluate a trained CNN checkpoint against an EchoFake split.

target = bonafide + replay_bonafide (genuine speaker, replayed or not)
nontarget = fake + replay_fake (spoofed, replayed or not)
This matches how the published EchoFake numbers are computed: replay
status makes detection harder, it doesn't change the ground-truth label.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.augment_gpu import random_crop_or_pad
from src.data.loaders import TARGET_SR, load_audio_from_bytes
from src.eval.metrics import compute_eer
from src.models.cnn import SmallCNN

WINDOW_LEN = 64600
TARGET_LABELS = {"bonafide", "replay_bonafide"}
NONTARGET_LABELS = {"fake", "replay_fake"}


def load_echofake_split(name: str) -> pd.DataFrame:
    data_dir = Path("data/raw/heldout/echofake/data")
    shards = sorted(data_dir.glob(f"{name}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards found for split '{name}' in {data_dir}")
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


def evaluate_checkpoint_on_echofake(checkpoint_path: Path, split: str, device: torch.device) -> dict:
    model = SmallCNN().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    df = load_echofake_split(split)
    rng = np.random.default_rng(0)  # deterministic crop for eval, not augmentation

    scores, labels = [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            if row["label"] not in TARGET_LABELS | NONTARGET_LABELS:
                continue
            x = load_audio_from_bytes(row["path"]["bytes"], source_desc=row["utt_id"])
            x = random_crop_or_pad(x, WINDOW_LEN, rng)
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)
            out = model(x_t)
            scores.append(out[0, 1].item())
            labels.append(1 if row["label"] in TARGET_LABELS else 0)

    scores = np.array(scores)
    labels = np.array(labels)
    eer, threshold = compute_eer(scores[labels == 1], scores[labels == 0])

    return {
        "split": split,
        "n_target": int((labels == 1).sum()),
        "n_nontarget": int((labels == 0).sum()),
        "eer": eer,
        "threshold": threshold,
        "checkpoint_epoch": ckpt["epoch"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, choices=["train", "dev", "closed_set_eval", "open_set_eval"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = evaluate_checkpoint_on_echofake(Path(args.checkpoint), args.split, device)
    print(
        f"split={result['split']} n_target={result['n_target']} n_nontarget={result['n_nontarget']} "
        f"EER={result['eer']*100:.2f}% (checkpoint epoch {result['checkpoint_epoch']})"
    )
