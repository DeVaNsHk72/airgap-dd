"""Stress tests for the one-class liveness model. Run from the airgap-dd root after
liveness_features.py:  PYTHONPATH=. python ../../paper/scripts/liveness_oneclass_checks.py"""
import csv, json, pickle, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from demo.pi_bundle import features as pi
from src.data.loaders import load_audio
from src.eval.metrics import compute_eer
warnings.filterwarnings("ignore")
D = np.load("../../paper/results/liveness_features.npz")
kind, split, y, path = D["kind"], D["split"], D["label"], D["path"]
tr, dv, te = split == "train", split == "dev", split == "test"
eer = lambda p, q: compute_eer(p, q)[0] * 100
def X(c):
    M = D["X_" + c].copy(); M[y == 1] = D["X_clean"][y == 1]; return M

class OC:
    def __init__(s, A):
        s.med = np.nanmedian(A, 0); A = s.imp(A); s.sc = StandardScaler().fit(A); s.c = LedoitWolf().fit(s.sc.transform(A))
    def imp(s, A): return np.where(np.isnan(A), s.med, A)
    def __call__(s, A): return -s.c.mahalanobis(s.sc.transform(s.imp(A)))

def clean_feat(p):
    x = load_audio(Path(p))
    return pi.feature_vector(x) if len(x) >= pi.CLIP_SAMPLES else None

if __name__ == "__main__":
    R = {}
    enroll = D["X_clean"][tr & (y == 1)]
    oc = OC(enroll)
    # threshold without any negative data: 5th percentile of cross-validated enrollment scores
    cv = np.concatenate([OC(enroll[a])(enroll[b]) for a, b in KFold(5, shuffle=True, random_state=0).split(enroll)])
    thr = float(np.percentile(cv, 5))
    live_te, live_dv = oc(D["X_clean"][te & (y == 1)]), oc(D["X_clean"][dv & (y == 1)])
    R["threshold_rule"] = "5th percentile of 5-fold cross-validated enrollment-day scores"
    R["live_accept_test_day"] = float(np.mean(live_te >= thr) * 100)
    R["live_accept_dev_day"] = float(np.mean(live_dv >= thr) * 100)
    for c in ("S", "R", "spk", "room"):
        neg = oc(X(c)[te & (y == 0)]); k = kind[te & (y == 0)]
        R["cond_" + c] = {"eer_test_day_live": eer(live_te, neg), "eer_dev_day_live": eer(live_dv, neg),
                          "reject": {kk: float(np.mean(neg[k == kk] < thr) * 100) for kk in sorted(set(k))}}
    # clones that are NOT replayed (digital injection, outside the threat model)
    rows = [r for r in csv.DictReader(open("data/demo/fakes/manifest.csv", encoding="utf-8")) if r["split"] == "test"]
    with ProcessPoolExecutor(8) as ex:
        F = list(ex.map(clean_feat, [r["path"].replace("\\", "/") for r in rows]))
    gen = np.array([r["generator"] for r, f in zip(rows, F) if f is not None]); F = np.stack([f for f in F if f is not None])
    s = oc(F)
    R["unreplayed_clones_oneclass"] = {g: {"eer": eer(live_te, s[gen == g]), "reject": float(np.mean(s[gen == g] < thr) * 100)} for g in sorted(set(gen))}
    base = pickle.load(open("checkpoints/lgbm_same_speaker_augmented.pkl", "rb")).booster_
    XS = X("S")
    P = {"objective": "binary", "num_leaves": 31, "learning_rate": 0.05, "seed": 0, "is_unbalance": True, "verbose": -1, "feature_pre_filter": False}
    dtr = lgb.Dataset(XS[tr], y[tr], free_raw_data=False)
    sup = lgb.train(P, dtr, 400, init_model=base, valid_sets=[lgb.Dataset(XS[dv], y[dv], reference=dtr)], callbacks=[lgb.early_stopping(40, verbose=False)])
    R["unreplayed_clones_supervised"] = {g: eer(sup.predict(D["X_clean"][te & (y == 1)]), sup.predict(F[gen == g])) for g in sorted(set(gen))}
    # enrollment size for the one-class model
    R["enroll_size"] = {}
    negS, negR = X("S")[te & (y == 0)], X("R")[te & (y == 0)]
    for n in (20, 40, 80, 160, len(enroll)):
        a, b = [], []
        for sd in range(10):
            m = OC(enroll[np.random.default_rng([0, n, sd]).choice(len(enroll), n, replace=False)])
            lt = m(D["X_clean"][te & (y == 1)]); a.append(eer(lt, m(negS))); b.append(eer(lt, m(negR)))
        R["enroll_size"][n] = {"sim_mean": float(np.mean(a)), "sim_std": float(np.std(a)), "meas_mean": float(np.mean(b)), "meas_std": float(np.std(b))}
    # which features move most under replay, in enrollment standard deviations
    names = pi.FEATURE_NAMES
    z = (np.nanmean(oc.sc.transform(oc.imp(negS)), 0))
    top = np.argsort(-np.abs(z))[:8]
    R["largest_mean_shift_sd"] = {names[i]: float(z[i]) for i in top}
    json.dump(R, open("../../paper/results/liveness_oneclass_checks.json", "w"), indent=1)
    print(json.dumps(R, indent=1))
