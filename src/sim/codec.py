"""Lossy codec round-trip: encode then decode through a real codec
implementation, so artifacts come from the actual codec, not an
approximation of one. Replay audio typically passes through a phone call,
a voice message, or a recording app's compressor before it's heard again.

AMR-NB and G.711 mu-law are telephony codecs designed for 8 kHz
narrowband speech — the bandwidth restriction itself is part of their
characteristic sound, so those two round-trip through 8 kHz. MP3 and
Opus both support 16 kHz natively and stay there.
"""

import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

CODECS = ["mp3", "opus", "amr_nb", "g711_mulaw"]
MP3_BITRATE_KBPS_RANGE = (32, 96)
OPUS_BITRATE_KBPS_RANGE = (16, 64)
AMR_NB_MODES_KBPS = [4.75, 5.15, 5.90, 6.70, 7.40, 7.95, 10.2, 12.2]


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run([FFMPEG_EXE, "-y", "-loglevel", "error", *args], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace"))


N_RETRIES = 3


def _roundtrip_once(x: np.ndarray, sr: int, codec_args: list[str], codec_sr: int, ext: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dry_wav = tmp / "dry.wav"
        coded = tmp / f"coded.{ext}"
        out_wav = tmp / "out.wav"

        sf.write(dry_wav, x, sr)
        _run_ffmpeg(["-i", str(dry_wav), "-ar", str(codec_sr), *codec_args, str(coded)])
        _run_ffmpeg(["-i", str(coded), "-ar", str(sr), str(out_wav)])

        y, _ = sf.read(out_wav, dtype="float32")
        return y


def _roundtrip(x: np.ndarray, sr: int, codec_args: list[str], codec_sr: int, ext: str) -> np.ndarray:
    """Real-world file I/O under heavy concurrent load (16+ worker
    processes all creating/deleting temp files) occasionally hits
    transient failures — e.g. antivirus briefly locking a freshly-written
    file — that a fresh attempt in a new temp dir resolves. Retried a few
    times before giving up, since a single flaky file operation shouldn't
    be treated the same as a deterministic bug.
    """
    last_error = None
    for attempt in range(N_RETRIES):
        try:
            return _roundtrip_once(x, sr, codec_args, codec_sr, ext)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"codec round-trip failed after {N_RETRIES} attempts: {last_error}") from last_error


def random_codec_roundtrip(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    codec = CODECS[rng.integers(len(CODECS))]

    if codec == "mp3":
        bitrate = rng.integers(*MP3_BITRATE_KBPS_RANGE)
        y = _roundtrip(x, sr, ["-c:a", "libmp3lame", "-b:a", f"{bitrate}k"], sr, "mp3")
    elif codec == "opus":
        bitrate = rng.integers(*OPUS_BITRATE_KBPS_RANGE)
        y = _roundtrip(x, sr, ["-c:a", "libopus", "-b:a", f"{bitrate}k"], sr, "opus")
    elif codec == "amr_nb":
        mode = AMR_NB_MODES_KBPS[rng.integers(len(AMR_NB_MODES_KBPS))]
        y = _roundtrip(x, sr, ["-c:a", "libopencore_amrnb", "-b:a", f"{mode}k", "-ac", "1"], 8000, "amr")
    else:  # g711_mulaw
        y = _roundtrip(x, sr, ["-c:a", "pcm_mulaw"], 8000, "wav")

    # codec round-trips can change length by a few samples (frame padding);
    # truncate/pad back to the original so downstream length assumptions hold
    if len(y) >= len(x):
        y = y[: len(x)]
    else:
        y = np.pad(y, (0, len(x) - len(y)))
    return y.astype(np.float32)
