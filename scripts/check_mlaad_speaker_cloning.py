"""Does MLAAD clone the source M-AILABS speaker's voice, or synthesize in
a generic/fixed voice? Pitch is a cheap decisive proxy: elliot_miller is
male (~110 Hz), LJSpeech-style default voices are female (~200 Hz).

If the cloning-capable models match the source speaker's pitch, we have
same-speaker genuine/clone pairs in training data. If not, a
speaker-specific detector trained on this would cheat via speaker ID.
"""

import csv
from pathlib import Path

import librosa
import numpy as np

from src.data.loaders import TARGET_SR, load_audio

MAILABS = Path("data/raw/train_sources/mailabs")
MLAAD = Path("data/raw/train_sources/mlaad")


def median_f0(x: np.ndarray) -> float:
    f0 = librosa.yin(x, fmin=60, fmax=400, sr=TARGET_SR)
    f0 = f0[np.isfinite(f0)]
    return float(np.median(f0)) if f0.size else float("nan")


if __name__ == "__main__":
    # genuine reference pitch per M-AILABS speaker
    print("=== genuine M-AILABS speakers ===")
    genuine = {}
    for spk, sub in [
        ("elliot_miller", "en_US/by_book/male/elliot_miller"),
        ("judy_bieber", "en_US/by_book/female/judy_bieber"),
        ("mary_ann", "en_US/by_book/female/mary_ann"),
        ("elizabeth_klett", "en_UK/by_book/female/elizabeth_klett"),
    ]:
        wavs = [w for w in (MAILABS / sub).rglob("*.wav") if not w.name.startswith("._")][:8]
        f0s = [median_f0(load_audio(w)) for w in wavs]
        genuine[spk] = float(np.nanmedian(f0s))
        print(f"  {spk:18s} median F0 = {genuine[spk]:6.1f} Hz")

    # for cloning-capable models, check pitch of fakes derived from elliot_miller (male)
    print("\n=== MLAAD fakes derived from elliot_miller (male, ~110 Hz genuine) ===")
    for model in [
        "tts_models_multilingual_multi-dataset_xtts_v2",
        "tts_models_multilingual_multi-dataset_bark",
        "OpenVoiceV2",
        "tts_models_en_ljspeech_vits",
    ]:
        meta = MLAAD / model / "meta.csv"
        if not meta.exists():
            continue
        picked = []
        with open(meta, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="|"):
                if "elliot_miller" in row["original_file"]:
                    wav = MLAAD / model / Path(row["path"]).name
                    if wav.exists():
                        picked.append(wav)
                if len(picked) >= 6:
                    break
        if not picked:
            print(f"  {model[:45]:45s} (no elliot_miller-derived samples found)")
            continue
        f0s = [median_f0(load_audio(w)) for w in picked]
        print(f"  {model[:45]:45s} median F0 = {np.nanmedian(f0s):6.1f} Hz  (n={len(picked)})")
