"""Benchmark additional cloning-capable TTS systems.

Determines which are fast enough to generate clones for 107 speakers.
XTTS-v2 measured at 2.93 s/clip. Architectural diversity matters more
than count: your_tts is VITS-based and bark is autoregressive, so both
differ from XTTS more than xtts_v1.1 does.
"""

import os
import time
from pathlib import Path

os.environ["COQUI_TOS_AGREED"] = "1"

CANDIDATES = [
    "tts_models/multilingual/multi-dataset/your_tts",
    "tts_models/multilingual/multi-dataset/xtts_v1.1",
]

TEXT = "The morning light came slowly through the window and settled on the floor."

if __name__ == "__main__":
    from TTS.api import TTS

    ref = Path("data/processed/asv_clones/_references")
    refs = sorted(ref.glob("*.wav"))
    if not refs:
        raise SystemExit("no reference wavs found")
    ref_wav = str(refs[0])
    out = Path("data/processed/_bench")
    out.mkdir(parents=True, exist_ok=True)

    for name in CANDIDATES:
        try:
            t0 = time.time()
            tts = TTS(name).to("cuda")
            load_s = time.time() - t0

            # warm-up excluded from timing
            tts.tts_to_file(
                text=TEXT, speaker_wav=ref_wav, language="en",
                file_path=str(out / "warm.wav"),
            )

            t0 = time.time()
            n = 3
            for i in range(n):
                tts.tts_to_file(
                    text=TEXT, speaker_wav=ref_wav, language="en",
                    file_path=str(out / f"b{i}.wav"),
                )
            per = (time.time() - t0) / n
            total_h = 107 * 40 * per / 3600
            print(f"OK   {name}")
            print(f"     load={load_s:.0f}s  per_clip={per:.2f}s  "
                  f"4280 clips = {total_h:.1f} h", flush=True)
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__}: {str(e)[:140]}", flush=True)
