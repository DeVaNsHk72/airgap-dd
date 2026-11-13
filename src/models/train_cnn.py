"""Second model: the small raw-waveform CNN, trained on GPU.

Dataset size bounded by measured DataLoader throughput, not an arbitrary
choice: codec.py's ffmpeg subprocess calls are CPU-bound and dominate
per-sample cost regardless of GPU speed. 16 DataLoader workers measured
fastest (17.3 samples/sec) — going higher (24, 32) made things WORSE
(12.3, 9.7 samples/sec) from process-spawn contention, not better. At
17.3/s the full ~106k-utterance training set would take ~1.7 hours per
epoch, impractical for multiple epochs, so this uses a bounded sample
(20k train, 4k dev) the same way the LightGBM baseline did.

Checkpointed every epoch (model, optimizer, epoch, RNG state) per
CLAUDE.md section 10 — Windows Update reboots are a real pre-emption
risk on this machine, not a hypothetical one.
"""

import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data.augment_gpu import make_dataloader
from src.data.protocols import build_bonafide_manifest, build_fake_manifest
from src.eval.leakage_check import assert_no_heldout_leakage
from src.eval.metrics import compute_eer
from src.models.cnn import SmallCNN

N_TRAIN_PER_CLASS = 10000
N_DEV_PER_CLASS = 2000
NUM_WORKERS = 16  # empirically fastest — see module docstring
BATCH_SIZE = 32
N_EPOCHS = 80
LR = 1e-4
RUN_SEED = 0
EARLY_STOP_PATIENCE = 10  # epochs without dev EER improvement before stopping

CHECKPOINT_DIR = Path("checkpoints/cnn_v1")
LOG_PATH = Path("logs/train_cnn.log")


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_cnn")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    return logger


def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            out = model(batch_x)
            scores.append(out[:, 1].cpu().numpy())
            labels.append(batch_y.numpy())
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    eer, _ = compute_eer(scores[labels == 1], scores[labels == 0])
    model.train()
    return eer


def main() -> None:
    global CHECKPOINT_DIR, LOG_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="train with replay_prob=0.0 (no simulated channel) — the G5 sanity-check baseline",
    )
    args = parser.parse_args()

    replay_prob = 0.0 if args.clean else 1.0
    if args.clean:
        CHECKPOINT_DIR = Path("checkpoints/cnn_clean_baseline")
        LOG_PATH = Path("logs/train_cnn_clean.log")

    logger = setup_logging()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(RUN_SEED)
    random.seed(RUN_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"run_seed={RUN_SEED} device={device} replay_prob={replay_prob}")

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
    assert_no_heldout_leakage(train_utts)
    assert_no_heldout_leakage(dev_utts)
    logger.info(f"G4 PASS: {len(train_utts)} train + {len(dev_utts)} dev, no held-out paths")

    rir_paths = list(Path("data/raw/rir").rglob("*.wav"))
    train_dataset, train_loader = make_dataloader(
        train_utts, rir_paths, run_seed=RUN_SEED, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, replay_prob=replay_prob
    )
    _, dev_loader = make_dataloader(
        dev_utts, rir_paths, run_seed=RUN_SEED + 1, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, replay_prob=replay_prob
    )

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_dev_eer = float("inf")
    epochs_without_improvement = 0
    if args.resume:
        ckpt_path = CHECKPOINT_DIR / "latest.pt"
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        random.setstate(ckpt["python_rng_state"])
        start_epoch = ckpt["epoch"] + 1
        best_dev_eer = ckpt.get("best_dev_eer", float("inf"))
        epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        logger.info(f"resumed from epoch {ckpt['epoch']}, best_dev_eer={best_dev_eer*100:.2f}%")

    for epoch in range(start_epoch, N_EPOCHS):
        train_dataset.set_epoch(epoch)
        epoch_start = time.time()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        dev_eer = evaluate(model, dev_loader, device)
        elapsed = time.time() - epoch_start

        improved = dev_eer < best_dev_eer
        if improved:
            best_dev_eer = dev_eer
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        logger.info(
            f"epoch {epoch}: loss={total_loss/n_batches:.4f} dev_eer={dev_eer*100:.2f}% "
            f"time={elapsed:.0f}s{' (best)' if improved else ''}"
        )
        print(
            f"epoch {epoch}: loss={total_loss/n_batches:.4f} dev_eer={dev_eer*100:.2f}% "
            f"time={elapsed:.0f}s{' (best)' if improved else ''}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "dev_eer": dev_eer,
            "best_dev_eer": best_dev_eer,
            "epochs_without_improvement": epochs_without_improvement,
        }
        torch.save(checkpoint, CHECKPOINT_DIR / "latest.pt")
        if improved:
            torch.save(checkpoint, CHECKPOINT_DIR / "best.pt")

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            logger.info(
                f"early stopping: no improvement for {EARLY_STOP_PATIENCE} epochs "
                f"(best dev_eer={best_dev_eer*100:.2f}%)"
            )
            print(f"early stopping at epoch {epoch}, best dev_eer={best_dev_eer*100:.2f}%")
            break

    logger.info(f"training complete, best_dev_eer={best_dev_eer*100:.2f}%")


if __name__ == "__main__":
    main()
