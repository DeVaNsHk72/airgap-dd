# Pi liveness demo

Tells **Devansh speaking live into the mic** apart from **audio played through a loudspeaker**
(a deepfake of him, or any recording of him). Personalised from the project's general
detector (`lgbm_same_speaker_augmented`) on Devansh's voice.

## Files

| File | What it is |
|---|---|
| `model.txt` | Fine-tuned LightGBM model (text format, loads in any LightGBM 4.x) |
| `config.json` | Threshold, feature order, and held-out test results |
| `features.py` | The 56 features, identical to training |
| `live_detect.py` | Records from the mic and prints LIVE HUMAN or LOUDSPEAKER / DEEPFAKE |
| `requirements-pi.txt` | Python packages |

## Setup on the Pi (Raspberry Pi OS 64-bit)

```bash
sudo apt install -y libportaudio2 libsndfile1 python3-venv
python3 -m venv ~/lv && source ~/lv/bin/activate
pip install -r requirements-pi.txt
```

Copy this whole folder to the Pi, plug the 3.5 mm mic into the USB sound card, then:

```bash
python live_detect.py --list-devices      # note the USB card's index
python live_detect.py --device <index>    # press Enter, speak ~3 s
```

## Demo tips

- Speak 20-40 cm from the mic, at the same level as the training recordings.
- Play the deepfake from a phone or Bluetooth speaker at a similar distance.
- First prediction is slow (library warm-up); later ones are quicker.
- Do a dry run on the Pi before the demo: 5 live tries and 5 playback tries.
  If live speech is wrongly rejected, the USB card sounds different from the
  laptop input the model was trained on (see Limits).

## Held-out test (16 Oct session, never used in training)

| | General base model | Fine-tuned on Devansh |
|---|---|---|
| Overall EER | 45.9% | **8.2%** |
| Live speech accepted | 74.3% | **91.0%** |
| Real recording via loudspeaker, rejected | 12.3% | **98.1%** |
| XTTS-v2 deepfake via loudspeaker, rejected | 82.5% | **94.2%** |
| YourTTS deepfake via loudspeaker (generator unseen in training), rejected | 90.0% | **87.5%** |

## Limits (say these if asked)

- **Personalised to Devansh.** The only live voice in training is Devansh's.
  Other people speaking live were never seen, so they may be scored as
  LOUDSPEAKER. Use Devansh as the live speaker for the demo.

- It detects **loudspeaker playback**, not synthesis. A real recording of Devansh
  played through a speaker is also flagged. For a voice device that is the intended
  behaviour, since replay is an attack.
- Trained on recordings made through the laptop's mic input; the Pi uses a USB
  sound card. No Pi recordings were available before the deadline.
- The loudspeaker class is *simulated* (speaker, distortion, room). Real speakers
  were not recorded. Held-out test numbers in `config.json` are on simulated
  playback, so they are an upper bound on what to expect live.
