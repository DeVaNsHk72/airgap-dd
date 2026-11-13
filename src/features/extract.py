"""Orchestrates spectral + reverb + residual feature extraction across a
manifest, with a disk cache. DSP runs once per (manifest, augmentation
seed) combination; LightGBM training/hyperparameter sweeps reload the
cached feature table instead of recomputing.

seed=None means clean, unaugmented features (used for the baseline G5
comparison — CLAUDE.md's clean-trained row must actually be clean).
seed=int means every utterance gets one fixed augmented realization,
seeded per-utterance so re-running with the same seed reproduces exactly
the same features (rule 5).
"""

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.loaders import TARGET_SR, load_audio
from src.data.protocols import Utterance
from src.features.residual import extract_residual_features
from src.features.reverb import extract_reverb_features
from src.features.spectral import extract_spectral_features
from src.sim.channel import simulate_replay_channel

CACHE_DIR = Path("data/processed/features")


def _cache_key(utterances: list[Utterance], seed: int | None) -> str:
    paths = sorted(str(u.path) for u in utterances)
    digest = hashlib.sha256(("|".join(paths) + f"|seed={seed}").encode()).hexdigest()[:16]
    return f"features_{digest}_seed{seed}.parquet"


def _extract_one(args: tuple) -> dict:
    u, i, seed, rir_paths = args
    x = load_audio(u.path)
    if seed is not None:
        rng = np.random.default_rng([seed, i])
        x = simulate_replay_channel(x, TARGET_SR, rng, rir_paths)

    row = {}
    row.update(extract_spectral_features(x, TARGET_SR))
    row.update(extract_reverb_features(x, TARGET_SR))
    row.update(extract_residual_features(x))
    row["label"] = 1 if u.label == "bonafide" else 0
    row["path"] = str(u.path)
    return row


def extract_features_for_manifest(
    utterances: list[Utterance],
    seed: int | None,
    rir_paths: list[Path] | None = None,
    force: bool = False,
    n_workers: int | None = None,
) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / _cache_key(utterances, seed)
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    if seed is not None and rir_paths is None:
        raise ValueError("rir_paths required when seed is not None")

    n_workers = n_workers or min(32, os.cpu_count() or 1)
    tasks = [(u, i, seed, rir_paths) for i, u in enumerate(utterances)]

    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for row in tqdm(
            pool.map(_extract_one, tasks, chunksize=8),
            total=len(tasks),
            desc=f"extracting features (seed={seed})",
        ):
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(cache_path)
    return df
