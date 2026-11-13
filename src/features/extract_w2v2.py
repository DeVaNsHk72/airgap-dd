"""Extract frozen W2V2 embeddings for a manifest, with a disk cache.

CPU (audio load + optional replay simulation) runs in a process pool;
GPU (W2V2 forward) runs batched in the main process, since the model
lives on one device. Same cache-key scheme as the hand-crafted feature
pipeline so both can be swapped freely.
"""

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data.loaders import TARGET_SR, load_audio
from src.data.protocols import Utterance
from src.sim.channel import simulate_replay_channel

CACHE_DIR = Path("data/processed/w2v2")
GPU_BATCH = 16


def _cache_key(utterances: list[Utterance], seed: int | None, resample_match: bool = False) -> str:
    paths = sorted(str(u.path) for u in utterances)
    digest = hashlib.sha256(("|".join(paths) + f"|w2v2|seed={seed}|rm={resample_match}").encode()).hexdigest()[:16]
    return f"w2v2_{digest}_seed{seed}.npz"


def _load_one(args: tuple) -> np.ndarray:
    u, i, seed, rir_paths, resample_match = args
    x = load_audio(u.path)

    # Format control: XTTS emits 24 kHz which we downsample to 16 kHz,
    # while genuine corpora are natively 16 kHz — a ~3 dB difference in
    # the 7-8 kHz band. Routing genuine audio through the same
    # 16k->24k->16k path equalises resampling history across classes.
    # Passed as data, not a patched global: Windows spawns workers, which
    # re-import modules fresh and would silently ignore a monkeypatch.
    if resample_match and "asv_clones" not in str(u.path) and "clones" not in str(u.path):
        import librosa

        up = librosa.resample(x, orig_sr=TARGET_SR, target_sr=24000)
        x = librosa.resample(up, orig_sr=24000, target_sr=TARGET_SR)

    if seed is not None:
        rng = np.random.default_rng([seed, i])
        try:
            x = simulate_replay_channel(x, TARGET_SR, rng, rir_paths)
        except Exception:
            pass  # fall back to unaugmented; logged upstream in augment_gpu
    return x.astype(np.float32)


def extract_w2v2_for_manifest(
    utterances: list[Utterance],
    seed: int | None,
    rir_paths: list[Path] | None = None,
    force: bool = False,
    n_workers: int | None = None,
    resample_match: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / _cache_key(utterances, seed, resample_match)
    if cache_path.exists() and not force:
        d = np.load(cache_path)
        return d["X"], d["y"]

    if seed is not None and rir_paths is None:
        raise ValueError("rir_paths required when seed is not None")

    from src.features.w2v2 import embed_batch

    n_workers = n_workers or min(16, os.cpu_count() or 1)
    tasks = [(u, i, seed, rir_paths, resample_match) for i, u in enumerate(utterances)]

    embeddings, buffer = [], []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for wav in tqdm(
            pool.map(_load_one, tasks, chunksize=4),
            total=len(tasks),
            desc=f"w2v2 embed (seed={seed})",
        ):
            buffer.append(wav)
            if len(buffer) >= GPU_BATCH:
                embeddings.append(embed_batch(buffer))
                buffer = []
    if buffer:
        embeddings.append(embed_batch(buffer))

    X = np.concatenate(embeddings, axis=0)
    y = np.array([1 if u.label == "bonafide" else 0 for u in utterances])
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y
