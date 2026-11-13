"""Evaluate a W2V2-embedding classifier against an EchoFake split.

Same target/nontarget convention as the other EchoFake evaluators:
target = bonafide + replay_bonafide, nontarget = fake + replay_fake.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data.loaders import load_audio_from_bytes
from src.eval.metrics import compute_eer
from src.eval.run_eval_echofake import NONTARGET_LABELS, TARGET_LABELS, load_echofake_split

GPU_BATCH = 16
CACHE_DIR = Path("data/processed/w2v2")


def embed_echofake(split: str, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"echofake_{split}.npz"
    if cache.exists() and not force:
        d = np.load(cache)
        return d["X"], d["y"]

    from src.features.w2v2 import embed_batch

    df = load_echofake_split(split)
    df = df[df["label"].isin(TARGET_LABELS | NONTARGET_LABELS)].reset_index(drop=True)

    embeddings, buffer, labels = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"w2v2 embed EchoFake {split}"):
        x = load_audio_from_bytes(row["path"]["bytes"], source_desc=row["utt_id"])
        buffer.append(x.astype(np.float32))
        labels.append(1 if row["label"] in TARGET_LABELS else 0)
        if len(buffer) >= GPU_BATCH:
            embeddings.append(embed_batch(buffer))
            buffer = []
    if buffer:
        embeddings.append(embed_batch(buffer))

    X = np.concatenate(embeddings, axis=0)
    y = np.array(labels)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", required=True, choices=["train", "dev", "closed_set_eval", "open_set_eval"])
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    X, y = embed_echofake(args.split)
    scores = model.predict_proba(X)[:, 1]
    eer, _ = compute_eer(scores[y == 1], scores[y == 0])
    print(f"split={args.split} n_target={(y==1).sum()} n_nontarget={(y==0).sum()} EER={eer*100:.2f}%")
