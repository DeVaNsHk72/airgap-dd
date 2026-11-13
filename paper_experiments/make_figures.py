"""Regenerate the paper figures from saved project results. Run from the airgap-dd root."""
import json, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from src.sim.nonlinear import random_nonlinear_distortion

OUT = "../../paper/figures/"
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3})
B, R = "#3b76af", "#c8553d"

# Fig: harmonics of a 440 Hz tone through the distortion stage
sr = 16000
t = np.arange(sr) / sr
x = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
f = np.fft.rfftfreq(sr, 1 / sr)
fig, ax = plt.subplots(figsize=(3.5, 2.2))
win = np.hanning(sr)
for seed, c in zip((1, 2, 3), ("#c8553d", "#e09f3e", "#6a4c93")):
    y = random_nonlinear_distortion(x, np.random.default_rng(seed))
    ax.semilogy(f, np.abs(np.fft.rfft(y * win)) + 1e-6, lw=0.8, color=c, label=f"distorted, seed {seed}")
ax.semilogy(f, np.abs(np.fft.rfft(x * win)) + 1e-6, lw=1.0, color="k", label="input tone")
for h in (880, 1760):
    ax.axvline(h, color="gray", ls=":", lw=0.6)
ax.set_xlim(0, 3000); ax.set_ylim(1e-4, 1e5)
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Magnitude")
ax.legend(fontsize=6.5, loc="upper right")
fig.tight_layout(); fig.savefig(OUT + "harmonics.pdf"); plt.close(fig)

# Fig: simulator vs ReplayDF (three validated descriptors)
pool = np.load("results/sim_demos/pool_metrics_corrected.npy", allow_pickle=True)
rdf = np.load("results/sim_demos/replaydf_metrics.npy", allow_pickle=True)
sim_t = np.load("results/sim_demos/sim_transfer_metrics.npy")
real_t = np.load("results/sim_demos/real_transfer_metrics.npy")
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.1))
panels = [
    (np.array([d["c50"] for d in pool]), np.array([d["c50"] for d in rdf]), np.linspace(-20, 40, 37), "$C_{50}$ (dB)"),
    (sim_t[:, 0], real_t[:, 0], np.linspace(-8, 6, 37), "Spectral tilt (dB/kHz)"),
    (sim_t[:, 1], real_t[:, 1], np.linspace(-30, 15, 37), "Energy above 4 kHz (dB)"),
]
for ax, (s, r, bins, lab) in zip(axs, panels):
    s = s[np.isfinite(s)]; r = r[np.isfinite(r)]
    ax.hist(s, bins=bins, density=True, alpha=0.6, color=B, label=f"simulated (n={len(s)})")
    ax.hist(r, bins=bins, density=True, alpha=0.6, color=R, label=f"ReplayDF (n={len(r)})")
    ax.set_xlabel(lab); ax.legend(fontsize=6.5)
    print(lab, "sim median %.2f real median %.2f" % (np.median(s), np.median(r)))
axs[0].set_ylabel("Density")
fig.tight_layout(); fig.savefig(OUT + "sim_validation.pdf"); plt.close(fig)
t60 = np.array([d["t60"] for d in rdf]); print("ReplayDF T60 >5s: %.0f%%, max %.0f s" % (100 * np.mean(t60 > 5), t60.max()))

# Fig: feature importance of the headline model
names = json.load(open("demo/pi_bundle/config.json"))["feature_names"]
m = pickle.load(open("checkpoints/lgbm_same_speaker_augmented.pkl", "rb"))
booster = m.booster_ if hasattr(m, "booster_") else m
print("trees", booster.num_trees(), "features", booster.num_feature())
gain = booster.feature_importance(importance_type="gain"); gain = 100 * gain / gain.sum()
order = np.argsort(gain)[::-1][:15]
grp = lambda n: R if n in ("spectral_tilt", "hf_energy", "modulation_depth") else ("#6a4c93" if n.startswith("residual") else B)
fig, ax = plt.subplots(figsize=(3.5, 2.9))
ax.barh([names[i].replace("_", " ") for i in order][::-1], gain[order][::-1], color=[grp(names[i]) for i in order][::-1])
ax.set_xlabel("Share of total split gain (%)")
fig.tight_layout(); fig.savefig(OUT + "importance.pdf"); plt.close(fig)
g = {"spectral": 0, "reverb": 0, "residual": 0}
for n, v in zip(names, gain):
    g["reverb" if grp(n) == R else "residual" if n.startswith("residual") else "spectral"] += v
print(g, [names[i] for i in order[:5]])
