"""Additional liveness experiments for the paper. Reads the cache written by
liveness_features.py. Run from the airgap-dd root:
    PYTHONPATH=. python ../../paper/scripts/liveness_experiments.py
"""
import json
import pickle
import warnings

import lightgbm as lgb
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.eval.metrics import compute_eer

warnings.filterwarnings("ignore")
SEED = 0
D = np.load("../../paper/results/liveness_features.npz")
kind, split, label, path = D["kind"], D["split"], D["label"], D["path"]
NAMES = json.load(open("demo/pi_bundle/config.json"))["feature_names"]
GROUPS = {
    "spectral (50)": list(range(50)),
    "reverberation (3)": [50, 51, 52],
    "residual (3)": [53, 54, 55],
    "reverberation + residual (6)": [50, 51, 52, 53, 54, 55],
    "all (56)": list(range(56)),
}
PARAMS = {"objective": "binary", "num_leaves": 31, "learning_rate": 0.05, "seed": SEED,
          "is_unbalance": True, "verbose": -1, "feature_pre_filter": False}
BASE = pickle.load(open("checkpoints/lgbm_same_speaker_augmented.pkl", "rb")).booster_
R = {}


def X(cond):
    """Feature matrix with live rows clean and loudspeaker rows under `cond`."""
    M = D["X_" + cond].copy()
    M[label == 1] = D["X_clean"][label == 1]
    return M


def eer(s, y):
    return compute_eer(s[y == 1], s[y == 0])[0] * 100


def boot(s, y, n=2000):
    rng = np.random.default_rng(SEED)
    p, q = s[y == 1], s[y == 0]
    v = [compute_eer(rng.choice(p, len(p)), rng.choice(q, len(q)))[0] * 100 for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def fit(Xtr, ytr, Xdv, ydv, init=None, cols=None, seed=SEED):
    cols = list(range(56)) if cols is None else cols
    dtr = lgb.Dataset(Xtr[:, cols], ytr, free_raw_data=False)
    ddv = lgb.Dataset(Xdv[:, cols], ydv, reference=dtr)
    m = lgb.train({**PARAMS, "seed": seed}, dtr, num_boost_round=400, init_model=init, valid_sets=[ddv],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
    return lambda Z: m.predict(Z[:, cols]), m


tr, dv, te = split == "train", split == "dev", split == "test"
XS, XR = X("S"), X("R")
y = label

# ---- E1: base, fine-tuned, from scratch; simulated and measured rooms at test
print("== E1 main")
models = {"general": (lambda Z: BASE.predict(Z)),
          "personalized (continued boosting)": fit(XS[tr], y[tr], XS[dv], y[dv], init=BASE)[0],
          "personal, from scratch": fit(XS[tr], y[tr], XS[dv], y[dv])[0]}
R["E1"] = {}
for name, f in models.items():
    row = {}
    for cname, XT in (("sim_rooms", XS), ("measured_rooms", XR)):
        s = f(XT[te])
        lo, hi = boot(s, y[te])
        row[cname] = {"eer": eer(s, y[te]), "ci": [lo, hi]}
        for k in sorted(set(kind[te & (y == 0)])):
            m = (kind[te] == k) | (y[te] == 1)
            row[cname]["eer_" + k] = eer(s[m], y[te][m])
    # operating point from the dev-day EER threshold
    sd = f(XS[dv]); thr = compute_eer(sd[y[dv] == 1], sd[y[dv] == 0])[1]
    st = f(XS[te])
    row["live_accept"] = float(np.mean(st[y[te] == 1] >= thr) * 100)
    row["reject"] = {k: float(np.mean(st[kind[te] == k] < thr) * 100) for k in sorted(set(kind[te & (y == 0)]))}
    R["E1"][name] = row
    print(f"{name:36s} sim {row['sim_rooms']['eer']:5.2f} [{row['sim_rooms']['ci'][0]:.1f},{row['sim_rooms']['ci'][1]:.1f}]"
          f"  measured {row['measured_rooms']['eer']:5.2f} [{row['measured_rooms']['ci'][0]:.1f},{row['measured_rooms']['ci'][1]:.1f}]"
          f"  live acc {row['live_accept']:.1f}  rej {row['reject']}")
    print("    per class sim:", {k: round(v, 2) for k, v in row["sim_rooms"].items() if k.startswith("eer_")})

# ---- E3: feature groups (from scratch)
print("== E3 feature groups")
R["E3"] = {}
for g, cols in GROUPS.items():
    f, _ = fit(XS[tr], y[tr], XS[dv], y[dv], cols=cols)
    R["E3"][g] = {"sim_rooms": eer(f(XS[te]), y[te]), "measured_rooms": eer(f(XR[te]), y[te])}
    print(f"{g:32s} sim {R['E3'][g]['sim_rooms']:5.2f}  measured {R['E3'][g]['measured_rooms']:5.2f}")

# ---- E5: which simulator stage is needed in training (continued boosting)
print("== E5 training channel")
R["E5"] = {}
for cond, desc in (("spk", "loudspeaker + distortion"), ("room", "room only"), ("S", "full chain")):
    XC = X(cond)
    f, _ = fit(XC[tr], y[tr], XC[dv], y[dv], init=BASE)
    R["E5"][desc] = {"sim_rooms": eer(f(XS[te]), y[te]), "measured_rooms": eer(f(XR[te]), y[te]),
                     "spk_only_test": eer(f(X("spk")[te]), y[te]), "room_only_test": eer(f(X("room")[te]), y[te])}
    print(f"{desc:28s}", {k: round(v, 2) for k, v in R["E5"][desc].items()})

# ---- E6: one-class models, live speech of the training day only
print("== E6 one-class")
R["E6"] = {}
for fitname, fm in (("train day", tr & (y == 1)), ("train + dev days", (tr | dv) & (y == 1))):
    for g in ("all (56)", "spectral (50)", "reverberation + residual (6)"):
        cols = GROUPS[g]
        # residual kurtosis is undefined for a few clips; impute with the enrollment median
        med = np.nanmedian(D["X_clean"][fm][:, cols], axis=0)
        imp = lambda A: np.where(np.isnan(A), med, A)
        sc = StandardScaler().fit(imp(D["X_clean"][fm][:, cols]))
        Z = sc.transform(imp(D["X_clean"][fm][:, cols]))
        oc = {
            "Mahalanobis": (lambda A, c=LedoitWolf().fit(Z): -c.mahalanobis(A)),
            "Isolation forest": (lambda A, m=IsolationForest(n_estimators=300, random_state=SEED).fit(Z): m.score_samples(A)),
            "One-class SVM": (lambda A, m=OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(Z): m.decision_function(A)),
        }
        for n, f in oc.items():
            key = f"{n} | {g} | {fitname}"
            row = {}
            for cname, XT in (("sim_rooms", XS), ("measured_rooms", XR), ("spk_only", X("spk")), ("room_only", X("room"))):
                s = f(sc.transform(imp(XT[te][:, cols])))
                row[cname] = eer(s, y[te])
                if cname in ("sim_rooms", "measured_rooms"):
                    row[cname + "_ci"] = boot(s, y[te], 1000)
                if cname == "sim_rooms":
                    for k in sorted(set(kind[te & (y == 0)])):
                        m = (kind[te] == k) | (y[te] == 1)
                        row["eer_" + k] = eer(s[m], y[te][m])
            R["E6"][key] = row
            print(f"{key:70s}", {k: (round(v, 2) if not isinstance(v, tuple) else [round(a, 1) for a in v]) for k, v in row.items()})

# ---- E7: how much enrollment speech is needed (continued boosting, 10 draws each)
print("== E7 enrollment size")
R["E7"] = {}
live_tr = np.where(tr & (kind == "live"))[0]
fake_tr = np.where(tr & (kind == "replay_xtts_v2"))[0]
rg_by_path = {p: i for i, p in enumerate(path) if kind[i] == "replay_genuine"}
for n in (10, 20, 40, 80, 160, len(live_tr)):
    vals = []
    for s in range(10):
        rng = np.random.default_rng([SEED, n, s])
        L = rng.choice(live_tr, n, replace=False)
        G = np.array([rg_by_path[path[i]] for i in L])
        F = rng.choice(fake_tr, max(1, round(n * len(fake_tr) / len(live_tr))), replace=False)
        idx = np.concatenate([L, G, F])
        f, _ = fit(XS[idx], y[idx], XS[dv], y[dv], init=BASE, seed=s)
        vals.append(eer(f(XS[te]), y[te]))
        if n == len(live_tr):
            break
    R["E7"][str(n)] = {"seconds": n * 2.5, "mean": float(np.mean(vals)), "std": float(np.std(vals)), "runs": len(vals)}
    print(f"n={n:4d} ({n * 2.5 / 60:4.1f} min)  EER {np.mean(vals):5.2f} +- {np.std(vals):.2f}")

R["counts"] = {s: {k: int(np.sum((split == s) & (kind == k))) for k in sorted(set(kind))} for s in ("train", "dev", "test")}
json.dump(R, open("../../paper/results/liveness_experiments.json", "w"), indent=1, default=float)
print(R["counts"])
