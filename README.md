# AirGap-ADD

Simulated acoustic channels for replay-robust, far-field audio deepfake detection.

**Status:** experiments complete. The primary hypothesis was tested and the result is negative.

## The question

Audio deepfake detectors are benchmarked on clean digital audio. A detector on a
physical device hears audio that went through a loudspeaker, a room and a microphone.
Published detectors that score below 5% equal error rate (EER) on clean audio score
42.9% to 46.3% on replayed audio.

The usual answer is to record real replay data. This project asks whether the replay
channel can be **simulated** well enough that a detector trained only on simulation
generalises to real re-recorded audio.

## Results

| Experiment | EER |
|---|---|
| Simulation-only training, EchoFake open set (target was below 43%) | 46.71% |
| Same model without the simulated channel | 50.92% |
| Per-speaker detector, generator seen in training | 12.00% |
| Per-speaker detector, five unseen generators | 53.14% |
| Personal liveness model, supervised, simulated playback | 5.55% |
| Personal liveness model, one-class (Mahalanobis), simulated playback | 1.82% |

The liveness results use one speaker and simulated playback only. Details and the
evidence for every number are under `results/` and `paper_experiments/results/`.

## Layout

```
src/sim/             replay channel simulator (loudspeaker, distortion, room, codec)
src/features/        spectral, reverberation and linear-prediction residual features
src/models/          LightGBM and small CNN detectors
src/eval/            EER implementation and evaluation harness
src/data/            loaders, protocols, clone generation
scripts/             training, evaluation and validation scripts
demo/                personalised liveness detector for a Raspberry Pi
paper_experiments/   scripts and result files for the liveness experiments in the paper
results/             validation reports and figures
tests/               unit tests
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

No audio is included. The corpora (M-AILABS, MLAAD, EchoFake, ReplayDF, ASVspoof 2019,
OpenSLR-28, MIT IR Survey) are available from their original sources; see
[DATA_INVENTORY.md](DATA_INVENTORY.md). The voice recordings of the enrolled speaker
used in `demo/` and `paper_experiments/` are not released, and trained model
checkpoints are not committed.

## Licence

Code MIT. Datasets follow their own licences; ReplayDF, EchoFake, IndicSynth and MLAAD
are non-commercial research releases and are not redistributed here.
