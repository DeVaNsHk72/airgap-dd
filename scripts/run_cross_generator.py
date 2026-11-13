"""The deciding experiment: per-speaker detection against UNSEEN generators.

For each EchoFake speaker:
  train on  = that speaker's genuine audio  vs  our XTTS-v2 clones of them
  test on   = held-out genuine  vs  EchoFake's fakes of them made by
              StyleTTS2, LLaSA, F5-TTS, SpeechT5, OpenAudio-S1-mini

The detector never sees an EchoFake fake during training, so those five
generators are genuinely unseen. XTTS-v2 fakes are scored separately as
the "seen generator" reference point.

Replay balance: EchoFake genuine audio includes real replayed variants
while our clones are clean, so simulated replay is applied to ~half the
clones. Without that the model could learn "replayed => genuine".

DISCLOSURE: uses EchoFake genuine audio for cloning references and
per-speaker training. Deliberate — it matches deployment, where a device
does hold its owner's audio — but it departs from CLAUDE.md rule 1.
"""

import collections
import csv
import glob
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.loaders import TARGET_SR, load_audio, load_audio_from_bytes
from src.eval.metrics import compute_eer
from src.features.w2v2 import embed_batch
from src.sim.channel import simulate_replay_channel

SPLIT = "open_set_eval"
MIN_GENUINE = 12
MIN_FAKES = 6
N_TEST_GENUINE = 6
CLONE_REPLAY_PROB = 0.5
GPU_BATCH = 16
SEEN_GENERATOR = "XTTS-v2"


def embed(waves):
    out = []
    for i in range(0, len(waves), GPU_BATCH):
        out.append(embed_batch(waves[i : i + GPU_BATCH]))
    return np.concatenate(out, axis=0) if out else np.empty((0, 1536), dtype=np.float32)


def load_echofake():
    files = sorted(glob.glob(f"data/raw/heldout/echofake/data/{SPLIT}-*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main(clone_manifests=None, tag="XTTS-only"):
    rng = random.Random(0)
    rir_paths = list(Path("data/raw/rir").rglob("*.wav"))

    if clone_manifests is None:
        clone_manifests = ["data/processed/echofake_clones/manifest.csv"]
    clones_by_spk = collections.defaultdict(list)
    for mpath in clone_manifests:
        if not Path(mpath).exists():
            print(f"  (missing {mpath}, skipping)", flush=True)
            continue
        with open(mpath, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="|"):
                p = Path(row["path"])
                if p.exists():
                    clones_by_spk[row["speaker"]].append(p)
    n_cl = sum(len(v) for v in clones_by_spk.values())
    print(f"[{tag}] {len(clone_manifests)} generator(s), {n_cl} clones", flush=True)

    df = load_echofake()
    bona = df[df["label"].isin(["bonafide", "replay_bonafide"])]
    fakes = df[df["label"].isin(["fake", "replay_fake"])]

    fbr = collections.defaultdict(list)
    for _, r in fakes.iterrows():
        ref = r["synthesis_details"].get("reference_speaker_id")
        if ref:
            fbr[ref].append(r)
    bbs = collections.defaultdict(list)
    for _, r in bona.iterrows():
        if r["source_speaker_id"]:
            bbs[r["source_speaker_id"]].append(r)

    speakers = [
        s for s in bbs
        if s in fbr and s in clones_by_spk
        and len(bbs[s]) >= MIN_GENUINE and len(fbr[s]) >= MIN_FAKES
    ]
    print(f"{len(speakers)} speakers", flush=True)

    all_scores, all_labels, all_gen = [], [], []
    done = 0
    for spk in speakers:
        gen_clips = bbs[spk][:]
        rng.shuffle(gen_clips)
        test_gen, train_gen = gen_clips[:N_TEST_GENUINE], gen_clips[N_TEST_GENUINE:]
        if len(train_gen) < 4:
            continue

        # --- training set: genuine vs our clones (replay-balanced) ---
        tr_waves, tr_y = [], []
        for r in train_gen:
            tr_waves.append(load_audio_from_bytes(r["path"]["bytes"], source_desc=r["utt_id"]))
            tr_y.append(1)
        for i, p in enumerate(clones_by_spk[spk]):
            x = load_audio(p)
            if rng.random() < CLONE_REPLAY_PROB:
                try:
                    x = simulate_replay_channel(x, TARGET_SR, np.random.default_rng([777, i]), rir_paths)
                except Exception:
                    pass
            tr_waves.append(x)
            tr_y.append(0)

        X_tr = embed(tr_waves)
        y_tr = np.array(tr_y)
        if len(np.unique(y_tr)) < 2:
            continue

        model = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, random_state=0, verbose=-1)
        model.fit(X_tr, y_tr)

        # --- test set: held-out genuine vs EchoFake fakes of this speaker ---
        te_waves, te_y, te_gen = [], [], []
        for r in test_gen:
            te_waves.append(load_audio_from_bytes(r["path"]["bytes"], source_desc=r["utt_id"]))
            te_y.append(1)
            te_gen.append("genuine")
        for r in fbr[spk]:
            te_waves.append(load_audio_from_bytes(r["path"]["bytes"], source_desc=r["utt_id"]))
            te_y.append(0)
            te_gen.append(r["synthesis_details"].get("model") or "unknown")

        X_te = embed(te_waves)
        s = model.predict_proba(X_te)[:, 1]
        all_scores.extend(s.tolist())
        all_labels.extend(te_y)
        all_gen.extend(te_gen)

        done += 1
        if done % 20 == 0:
            print(f"  {done}/{len(speakers)} speakers done", flush=True)

    scores = np.array(all_scores)
    labels = np.array(all_labels)
    gens = np.array(all_gen)
    print(f"\ntrials: {len(labels)} ({(labels==1).sum()} genuine, {(labels==0).sum()} fake)")

    # unseen generators only
    mask_unseen = (labels == 1) | ((labels == 0) & (gens != SEEN_GENERATOR))
    eer_unseen, _ = compute_eer(scores[mask_unseen & (labels == 1)], scores[mask_unseen & (labels == 0)])
    print(f"\nCROSS-GENERATOR EER (5 unseen generators): {eer_unseen*100:.2f}%")

    mask_seen = (labels == 1) | ((labels == 0) & (gens == SEEN_GENERATOR))
    eer_seen, _ = compute_eer(scores[mask_seen & (labels == 1)], scores[mask_seen & (labels == 0)])
    print(f"[{tag}] same-generator EER (XTTS-v2, trained on): {eer_seen*100:.2f}%")

    print("\nper-generator:")
    for g in sorted(set(gens) - {"genuine"}):
        m = (labels == 1) | ((labels == 0) & (gens == g))
        e, _ = compute_eer(scores[m & (labels == 1)], scores[m & (labels == 0)])
        n = ((labels == 0) & (gens == g)).sum()
        seen_mark = "  (seen)" if g == SEEN_GENERATOR else ""
        print(f"  {g:22s} n={n:5d}  EER={e*100:6.2f}%{seen_mark}")


if __name__ == "__main__":
    import sys
    if "--multi" in sys.argv:
        main(
            clone_manifests=[
                "data/processed/echofake_clones/manifest.csv",
                "data/processed/echofake_clones_yourtts/manifest.csv",
            ],
            tag="XTTS+YourTTS",
        )
    else:
        main()
