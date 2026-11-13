"""Live vs loudspeaker detector for the Raspberry Pi demo.

  python live_detect.py --list-devices          # find the USB sound card
  python live_detect.py --device 1              # press Enter, then speak or play
  python live_detect.py --device 1 --continuous # score every window, no Enter
  python live_detect.py --wav some_file.wav     # score a file (no mic needed)

Score is P(live human). At or above the threshold in config.json -> LIVE,
below -> LOUDSPEAKER (deepfake or any playback).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import lightgbm as lgb
import numpy as np

import features as F

HERE = Path(__file__).resolve().parent
RECORD_SECONDS = 4.0  # a little longer than one clip, so trimming still leaves 2.5 s


def load_model() -> tuple[lgb.Booster, float]:
    cfg = json.loads((HERE / "config.json").read_text())
    assert cfg["feature_names"] == F.FEATURE_NAMES, "features.py does not match the trained model"
    return lgb.Booster(model_file=str(HERE / "model.txt")), float(cfg["threshold"])


def to_clip(x: np.ndarray, sr: int) -> np.ndarray | None:
    """Same path as training: 16 kHz mono, edge-trim, 2.5 s window, speech gate."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != F.SR:
        x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=F.SR)
    x, _ = librosa.effects.trim(x, top_db=F.TRIM_TOP_DB)
    if len(x) < F.CLIP_SAMPLES:
        return None
    clip = x[: F.CLIP_SAMPLES]
    if F.speech_fraction(clip) < F.MIN_SPEECH_FRACTION:
        return None
    return clip


def score(model: lgb.Booster, thr: float, x: np.ndarray, sr: int) -> None:
    t0 = time.perf_counter()
    clip = to_clip(x, sr)
    if clip is None:
        print("  (not enough speech, speak for about 3 seconds)")
        return
    p = float(model.predict(F.feature_vector(clip)[None, :])[0])
    ms = (time.perf_counter() - t0) * 1000
    verdict = "LIVE HUMAN" if p >= thr else "LOUDSPEAKER / DEEPFAKE"
    print(f"  {verdict:24s} score={p:.3f} (threshold {thr:.3f})  [{ms:.0f} ms]")


def record(device: int | None, sr: int) -> np.ndarray:
    import sounddevice as sd

    x = sd.rec(int(RECORD_SECONDS * sr), samplerate=sr, channels=1, dtype="float32", device=device)
    sd.wait()
    return x[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--rate", type=int, default=48000, help="capture rate; most USB cards do 48000")
    ap.add_argument("--continuous", action="store_true")
    ap.add_argument("--wav", type=Path)
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    model, thr = load_model()

    if args.wav:
        import soundfile as sf

        x, sr = sf.read(args.wav, dtype="float32")
        score(model, thr, x, sr)
        return

    print(f"ready. threshold={thr:.3f}. Ctrl+C to quit.")
    try:
        while True:
            if not args.continuous:
                input("\npress Enter, then speak (or play the deepfake) for 4 s... ")
            print("  recording...")
            score(model, thr, record(args.device, args.rate), args.rate)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
