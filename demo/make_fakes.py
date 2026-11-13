"""Deepfakes of Devansh's voice for the Pi liveness demo.

Each split clones from reference clips of its OWN session only, so the
held-out 16 Oct test never shares reference audio with training:

  train  XTTS-v2   refs from 2026-10-15
  dev    XTTS-v2   refs from 2026-09-08
  test   XTTS-v2   refs from 2026-10-16
  test   YourTTS   refs from 2026-10-16   <- generator never used in training

No speech recogniser is installed, so clones read everyday sentences rather
than Devansh's own words. The replayed-genuine negatives in finetune.py
carry his exact content, which keeps "different words" from being the cue.

Resume-safe: existing outputs are skipped.
"""

import csv
import os
import random
from pathlib import Path

os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS

VOICE_MANIFEST = Path("data/demo/my_voice/manifest.csv")
OUT_ROOT = Path("data/demo/fakes")
N_REFS = 6
SEED = 0

XTTS = "tts_models/multilingual/multi-dataset/xtts_v2"
YOURTTS = "tts_models/multilingual/multi-dataset/your_tts"

JOBS = [
    # (split, generator, model, ref session, count)
    ("train", "xtts_v2", XTTS, "2026-10-15", 300),
    ("dev", "xtts_v2", XTTS, "2026-09-08", 60),
    ("test", "xtts_v2", XTTS, "2026-10-16", 120),
    ("test", "your_tts", YOURTTS, "2026-10-16", 120),
]

SENTENCES = [
    "Hey, can you turn on the lights in the living room?",
    "What is the weather going to be like tomorrow morning?",
    "Please unlock the front door, I am almost home.",
    "Set an alarm for six thirty and remind me to call my mother.",
    "Play some music, something calm for studying.",
    "Transfer five thousand rupees to my savings account.",
    "Read out my latest messages, please.",
    "I am running late, tell the team the meeting starts at four.",
    "Turn the air conditioner down to twenty two degrees.",
    "Add milk, bread and eggs to the shopping list.",
    "What time does the train to the city leave tonight?",
    "Open the garage and switch off the porch light.",
    "Can you order the same food as last Friday?",
    "Call the bank and check if my card is still blocked.",
    "Remind me to submit the project report before midnight.",
    "This is Devansh, please approve the payment request.",
    "How long will it take to drive to the airport right now?",
    "Send a message to Rahul saying I will be there in ten minutes.",
    "Lower the volume and pause the video for a moment.",
    "Book a table for two at eight o'clock this evening.",
    "Disable the security camera in the hallway.",
    "Tell me the news headlines for today.",
    "I forgot my password, can you reset it for me?",
    "Start the washing machine on the quick cycle.",
    "Is there anything on my calendar for Saturday afternoon?",
    "Okay, I think that is everything for now, thank you.",
    "My voice is my password, verify me.",
    "Please share the location with my brother.",
    "Order a cab to the railway station.",
    "Change the thermostat schedule for the weekend.",
    "I would like to check my account balance.",
    "Can you explain how this works one more time?",
    "Let me know when the delivery arrives at the gate.",
    "Turn off everything in the house, I am going to sleep.",
    "We need to finish the demonstration before the deadline.",
    "Hello, this is a quick test of the microphone.",
    "Please confirm the order and pay with my saved card.",
    "Show me the photos I took yesterday at the park.",
    "Write down a note: buy a new charger tomorrow.",
    "Good morning, what do I have planned for today?",
]


def session_refs(session: str, rng: random.Random) -> list[str]:
    with open(VOICE_MANIFEST, newline="") as f:
        clips = [r["path"] for r in csv.DictReader(f) if r["session"] == session]
    assert len(clips) >= N_REFS, f"only {len(clips)} clips in session {session}"
    return rng.sample(clips, N_REFS)


def main() -> None:
    rng = random.Random(SEED)
    manifest = OUT_ROOT / "manifest.csv"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    is_new = not manifest.exists()

    loaded: dict[str, TTS] = {}
    with open(manifest, "a", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        if is_new:
            w.writerow(["path", "split", "generator", "ref_session", "text"])

        for split, gen, model_name, ref_session, count in JOBS:
            refs = session_refs(ref_session, rng)
            if model_name not in loaded:
                loaded[model_name] = TTS(model_name).to("cuda")
            tts = loaded[model_name]
            out_dir = OUT_ROOT / split / gen
            out_dir.mkdir(parents=True, exist_ok=True)

            made = 0
            for i in range(count):
                # Two sentences per fake: one alone often trims to under 2.5 s.
                n = len(SENTENCES)
                text = f"{SENTENCES[i % n]} {SENTENCES[(i * 7 + 3) % n]}"
                out = out_dir / f"{gen}_{i:04d}.wav"
                if out.exists():
                    continue
                # XTTS accepts several references; YourTTS takes one.
                spk = refs if gen == "xtts_v2" else refs[i % len(refs)]
                tts.tts_to_file(text=text, speaker_wav=spk, language="en", file_path=str(out))
                w.writerow([str(out), split, gen, ref_session, text])
                mf.flush()
                made += 1
            print(f"{split}/{gen}: {made} new (target {count})", flush=True)


if __name__ == "__main__":
    main()
