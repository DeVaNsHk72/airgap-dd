"""Turn Devansh's raw voice memos into fixed-length 16 kHz clips for the
Raspberry Pi liveness demo (live speech vs loudspeaker playback).

This is demo data, not paper data: it never enters a paper table and it
lives under data/demo/, which is not committed (rule 6).

Steps per recording:
  1. ffmpeg decodes the .m4a to 16 kHz mono wav. The memos are 48 kHz
     stereo; load_audio asserts mono, so the downmix happens here.
  2. load_audio applies the same edge-trim every other corpus gets (rule 4).
  3. The trimmed audio is cut into non-overlapping 2.5 s clips. Fixed length
     because duration alone once separated classes at 6% EER (PROJECT.md 6.2).
  4. Clips under MIN_SPEECH_FRACTION speech are dropped, so the model learns
     voices rather than pauses.

Each clip is tagged with its session (recording date). Splits are made by
session later, never by clip, because clips from one recording share a room,
mic position and background noise.
"""

import argparse
import csv
import subprocess
from datetime import datetime
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf

from src.data.loaders import TARGET_SR, TRIM_TOP_DB, load_audio

CLIP_SECONDS = 2.5
CLIP_SAMPLES = int(TARGET_SR * CLIP_SECONDS)
MIN_SPEECH_FRACTION = 0.6

DEFAULT_SRC = Path.home() / "Documents" / "Sound Recordings"
DEFAULT_OUT = Path("data/demo/my_voice")


def decode_to_wav(src: Path, dst: Path) -> None:
    """Decode any ffmpeg-readable file to 16 kHz mono PCM wav."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-i", str(src), "-ac", "1", "-ar", str(TARGET_SR), "-c:a", "pcm_s16le", str(dst),
    ]
    subprocess.run(cmd, check=True)


def speech_fraction(x: np.ndarray) -> float:
    """Share of samples inside non-silent intervals, same threshold as the trim."""
    intervals = librosa.effects.split(x, top_db=TRIM_TOP_DB)
    voiced = sum(end - start for start, end in intervals)
    return voiced / len(x)


def cut_clips(x: np.ndarray) -> list[np.ndarray]:
    """Non-overlapping fixed-length clips; the short tail is discarded."""
    n = len(x) // CLIP_SAMPLES
    return [x[i * CLIP_SAMPLES : (i + 1) * CLIP_SAMPLES] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    recordings = sorted(args.src.glob("*.m4a"), key=lambda p: p.stat().st_mtime)
    assert recordings, f"no .m4a files in {args.src}"

    rows = []
    for rec in recordings:
        session = datetime.fromtimestamp(rec.stat().st_mtime).strftime("%Y-%m-%d")
        stem = rec.stem.replace(" ", "_").replace("(", "").replace(")", "")
        wav = args.out / "decoded" / f"{stem}.wav"
        decode_to_wav(rec, wav)

        x = load_audio(wav)
        kept = 0
        for i, clip in enumerate(cut_clips(x)):
            frac = speech_fraction(clip)
            if frac < MIN_SPEECH_FRACTION:
                continue
            clip_path = args.out / "clips" / session / f"{stem}_{i:04d}.wav"
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(clip_path, clip, TARGET_SR, subtype="PCM_16")
            rows.append({"path": str(clip_path), "label": "live", "session": session,
                         "source": rec.name, "speech_fraction": round(frac, 3)})
            kept += 1
        print(f"{rec.name:22s} session={session}  {len(x)/TARGET_SR:6.1f}s trimmed  -> {kept} clips")

    manifest = args.out / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by_session: dict[str, int] = {}
    for r in rows:
        by_session[r["session"]] = by_session.get(r["session"], 0) + 1
    print(f"\n{len(rows)} clips written, manifest at {manifest}")
    for s, n in sorted(by_session.items()):
        print(f"  {s}: {n} clips ({n * CLIP_SECONDS / 60:.1f} min)")


if __name__ == "__main__":
    main()
