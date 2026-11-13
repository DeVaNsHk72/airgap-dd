"""Positive control for the cross-generator negative result.

If a per-speaker detector cannot detect HELD-OUT clones from the generator
it trained on, the pipeline is broken and the 53.14% cross-generator
figure is meaningless. This holds out 25% of each speaker's XTTS clones
from training and scores them.
"""
import collections, csv, glob, random
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
from src.data.loaders import TARGET_SR, load_audio, load_audio_from_bytes
from src.eval.metrics import compute_eer
from src.features.w2v2 import embed_batch
from src.sim.channel import simulate_replay_channel

def embed(w):
    return np.concatenate([embed_batch(w[i:i+16]) for i in range(0, len(w), 16)], axis=0)

if __name__ == "__main__":
    rng = random.Random(0)
    rirs = list(Path("data/raw/rir").rglob("*.wav"))
    clones = collections.defaultdict(list)
    with open("data/processed/echofake_clones/manifest.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="|"):
            clones[r["speaker"]].append(Path(r["path"]))

    files = sorted(glob.glob("data/raw/heldout/echofake/data/open_set_eval-*.parquet"))
    df = pd.concat([pd.read_parquet(x) for x in files], ignore_index=True)
    bona = df[df["label"].isin(["bonafide","replay_bonafide"])]
    bbs = collections.defaultdict(list)
    for _, r in bona.iterrows():
        if r["source_speaker_id"]: bbs[r["source_speaker_id"]].append(r)

    spks = [s for s in bbs if s in clones and len(bbs[s]) >= 12][:30]
    S, Y = [], []
    for spk in spks:
        g = bbs[spk][:]; rng.shuffle(g)
        te_g, tr_g = g[:6], g[6:]
        c = clones[spk][:]; rng.shuffle(c)
        n_hold = max(4, len(c)//4)
        te_c, tr_c = c[:n_hold], c[n_hold:]

        tw, ty = [], []
        for r in tr_g: tw.append(load_audio_from_bytes(r["path"]["bytes"])); ty.append(1)
        for i,p in enumerate(tr_c):
            x = load_audio(p)
            if rng.random() < 0.5:
                try: x = simulate_replay_channel(x, TARGET_SR, np.random.default_rng([777,i]), rirs)
                except Exception: pass
            tw.append(x); ty.append(0)
        m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, random_state=0, verbose=-1)
        m.fit(embed(tw), np.array(ty))

        vw, vy = [], []
        for r in te_g: vw.append(load_audio_from_bytes(r["path"]["bytes"])); vy.append(1)
        for p in te_c: vw.append(load_audio(p)); vy.append(0)
        S.extend(m.predict_proba(embed(vw))[:,1].tolist()); Y.extend(vy)

    S, Y = np.array(S), np.array(Y)
    e, _ = compute_eer(S[Y==1], S[Y==0])
    print(f"\nPOSITIVE CONTROL — held-out XTTS clones (same generator as training)")
    print(f"speakers={len(spks)} trials={len(Y)}  EER={e*100:.2f}%")
    print("low = pipeline works, so 53.14% cross-generator is a real negative")
    print("~50% = pipeline broken, cross-generator result meaningless")
