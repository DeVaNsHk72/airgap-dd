"""Does clip duration alone separate genuine from clone?

ASVspoof genuine clips are short (median ~2.8s); clones synthesize
M-AILABS book text and run much longer. If duration alone classifies
well, the 6.24% result is contaminated by a trivial length cue rather
than measuring synthesis detection.
"""
import random
import numpy as np
from src.data.asv_clone_protocol import build_asv_clone_manifest
from src.data.asvspoof_protocol import build_asv_bonafide_manifest
from src.data.loaders import TARGET_SR, load_audio
from src.eval.metrics import compute_eer

if __name__ == "__main__":
    rng = random.Random(0)
    bona = rng.sample([u for u in build_asv_bonafide_manifest() if u.split == "dev"], 300)
    clones = rng.sample([u for u in build_asv_clone_manifest() if u.split == "dev"], 300)

    bd = np.array([len(load_audio(u.path)) / TARGET_SR for u in bona])
    cd = np.array([len(load_audio(u.path)) / TARGET_SR for u in clones])

    print(f"genuine duration: median={np.median(bd):.2f}s  IQR={np.percentile(bd,25):.2f}-{np.percentile(bd,75):.2f}")
    print(f"clone   duration: median={np.median(cd):.2f}s  IQR={np.percentile(cd,25):.2f}-{np.percentile(cd,75):.2f}")

    a, _ = compute_eer(bd, cd)
    b, _ = compute_eer(-bd, -cd)
    print(f"\nEER using DURATION ALONE: {min(a,b)*100:.2f}%")
    print("near 50% = no length cue; low = the 6.24% is contaminated")
