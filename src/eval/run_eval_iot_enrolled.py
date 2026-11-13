"""Evaluate the enrolled-speaker detector on EchoFake.

Uses EchoFake speakers who appear BOTH as bona fide and as the reference
voice for fakes cloned from them — the genuine IoT case: a device enrolled
to a person, attacked with a clone of that person.

DISCLOSURE: EchoFake supplies enrolment audio and evaluation trials here.
The model's weights are trained only on M-AILABS + our own XTTS clones —
no gradient ever touches EchoFake — but enrolment does feed held-out audio
to the system at inference. This deviates from a strict reading of
CLAUDE.md rule 1 and must be stated in any write-up. It follows standard
speaker-verification practice, where enrolment audio always comes from the
evaluation set, but the deviation is real and is not hidden.

Enrolment clips are disjoint from scored clips.
"""

import argparse
import collections
import pickle
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data.loaders import load_audio_from_bytes
from src.eval.metrics import compute_eer
from src.eval.run_eval_echofake import load_echofake_split
from src.features.enrollment import build_matrix, enrollment_centroid

N_ENROL = 3  # a device would hold only a handful of clips
MIN_TRIALS = 2  # skip speakers with too little data to be meaningful
GPU_BATCH = 16


def _embed(waves: list[np.ndarray]) -> np.ndarray:
    from src.features.w2v2 import embed_batch

    out = []
    for i in range(0, len(waves), GPU_BATCH):
        out.append(embed_batch(waves[i : i + GPU_BATCH]))
    return np.concatenate(out, axis=0)


def build_trials(split: str, seed: int = 0):
    """Per speaker: enrolment clips (genuine), plus scored trials of
    genuine (target) and clones-of-them (nontarget)."""
    rng = random.Random(seed)
    df = load_echofake_split(split)

    bona = df[df["label"].isin(["bonafide", "replay_bonafide"])]
    fakes = df[df["label"].isin(["fake", "replay_fake"])]

    # which speaker was cloned to make each fake
    fake_by_ref = collections.defaultdict(list)
    for _, r in fakes.iterrows():
        ref = r["synthesis_details"].get("reference_speaker_id")
        if ref:
            fake_by_ref[ref].append(r)

    bona_by_spk = collections.defaultdict(list)
    for _, r in bona.iterrows():
        if r["source_speaker_id"]:
            bona_by_spk[r["source_speaker_id"]].append(r)

    usable = [
        s
        for s in bona_by_spk
        if s in fake_by_ref
        and len(bona_by_spk[s]) >= N_ENROL + MIN_TRIALS
        and len(fake_by_ref[s]) >= MIN_TRIALS
    ]
    print(f"{len(usable)} speakers with enough enrolment + trial data")

    enrol_rows, trial_rows, trial_spk, trial_lab = [], [], [], []
    for spk in usable:
        clips = bona_by_spk[spk][:]
        rng.shuffle(clips)
        enrol = clips[:N_ENROL]
        rest = clips[N_ENROL:]
        for r in enrol:
            enrol_rows.append((spk, r))
        for r in rest:
            trial_rows.append(r)
            trial_spk.append(spk)
            trial_lab.append(1)
        for r in fake_by_ref[spk]:
            trial_rows.append(r)
            trial_spk.append(spk)
            trial_lab.append(0)
    return enrol_rows, trial_rows, trial_spk, np.array(trial_lab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="dev", choices=["dev", "closed_set_eval", "open_set_eval"])
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    enrol_rows, trial_rows, trial_spk, y = build_trials(args.split)
    print(f"trials: {len(trial_rows)} ({(y==1).sum()} genuine, {(y==0).sum()} clone)")

    enrol_waves = [
        load_audio_from_bytes(r["path"]["bytes"], source_desc=r["utt_id"])
        for _, r in tqdm(enrol_rows, desc="enrol audio")
    ]
    E = _embed(enrol_waves)
    centroids: dict[str, np.ndarray] = {}
    idx = 0
    by_spk_count = collections.Counter(spk for spk, _ in enrol_rows)
    for spk in dict.fromkeys(spk for spk, _ in enrol_rows):
        n = by_spk_count[spk]
        centroids[spk] = enrollment_centroid(E[idx : idx + n])
        idx += n

    trial_waves = [
        load_audio_from_bytes(r["path"]["bytes"], source_desc=r["utt_id"])
        for r in tqdm(trial_rows, desc="trial audio")
    ]
    T = _embed(trial_waves)

    X = build_matrix(T, np.stack([centroids[s] for s in trial_spk]))
    scores = model.predict_proba(X)[:, 1]
    eer, _ = compute_eer(scores[y == 1], scores[y == 0])
    print(f"\nsplit={args.split} speakers={len(centroids)} trials={len(y)} "
          f"IoT enrolled-speaker EER={eer*100:.2f}%")
