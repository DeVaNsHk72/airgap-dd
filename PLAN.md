# AirGap-ADD

**Simulated Acoustic Channels for Replay-Robust, Far-Field Audio Deepfake Detection**

Research plan. Target venues: ICASSP 2027, Interspeech 2027. Fallback: SPCOM / NCC / ASVspoof workshop.

---

## 1. The claim

Audio deepfake detectors are trained and benchmarked on digitally clean audio. Any detector that lives on a physical device never sees that. It hears sound that travelled through a loudspeaker, a room, and a cheap microphone.

That gap is not cosmetic. Müller et al. (Interspeech 2025) showed that simply playing a deepfake through a speaker and re-recording it pushes W2V2-AASIST from 4.7% EER to 18.2%. Replaying the audio launders away the synthesis artifacts the detector was trained to find.

Everyone's proposed fix is to record more replay data. That does not scale: there are thousands of microphone, speaker, room and distance combinations, and no lab can cover them.

**Our claim: you can simulate the replay channel well enough that a detector trained purely on simulation generalises to real re-recorded audio it has never seen.**

If true, replay robustness becomes free. Anyone can generate the training data on a GPU overnight.

### Why this is defensible novelty

| Existing work | What it did | What it left open |
|---|---|---|
| ReplayDF (Interspeech 2025) | Showed replay breaks detectors, released 132.5h real re-recordings | Did not propose a fix; models are large |
| EchoFake (2025) | Replay-aware dataset, 16 closed + 4 open conditions | Trains on real replay data; no simulation, no edge |
| Edge SSL browser plugin (arXiv 2606.30780) | Truncated SSL + linear head, beats AASIST by 10%, 40% faster | Evaluated on clean digital audio only |
| ASVspoof PA | Simulated replay detection | Replayed content is *real human* speech, not synthetic |

We sit exactly in the hole: **synthetic speech, replayed, detected by a small model, trained without any real replay data.**

Note on ASVspoof PA: its evaluation set was itself partly simulated, so the community already accepts channel simulation as valid. Nobody has applied it to the deepfake (LA/DF) problem.


### Numeric target

From EchoFake (arXiv 2510.19414) Table 3, evaluated on EchoFake open-set:

| Trained on | RawNet2 | AASIST | Wav2Vec2 |
|---|---|---|---|
| ASV19LA (clean only) | 46.34 | 43.23 | 42.94 |
| EchoFake-train (real replay, oracle) | 21.13 | 14.88 | 11.86 |

Published, so cite rather than re-run. Our contribution is the row between:
trained on simulation only. Below ~43% supports the hypothesis; near ~15% is a
strong result; at or above 43% means simulation did not transfer, which is an
honest finding worth reporting.

### Secondary contributions

1. **Quantisation interacts with replay.** Nobody has measured whether INT8 and distilled models degrade *worse* under replay than their FP32 teachers. Cheap to test, novel either way.
2. **Three-class formulation.** Binary real/fake breaks in this setting because bona fide audio is also re-recorded. We use live-human / replayed-human / replayed-synthetic.
3. **Unseen-generator generalisation.** Channel cues are independent of the TTS system. Hypothesis: our detector's generalisation gap to held-out generators is smaller than an artifact-based detector's.

---

## 2. Kill criterion

**Run this in week 1. Do not proceed if it fails.**

Take 100 clips (50 real, 50 fake). Score with pretrained AASIST, record EER. Play them out of a laptop speaker, record with a phone at 1m, score again.

- EER jump of 8 points or more: proceed.
- Jump under 3 points: the premise is wrong for your setup. Stop and re-scope.

Log the result in `results/week1_pilot.md` either way. It becomes Figure 1.

---

## 3. Resources

### 3.1 Papers to read before writing code

| Paper | Link | Why |
|---|---|---|
| Replay Attacks Against Audio Deepfake Detection | arxiv.org/abs/2505.14862 | The effect we build on |
| EchoFake | arxiv.org/abs/2510.19414 | Closest competitor, must differentiate |
| AASIST | ICASSP 2022, code at github.com/clovaai/aasist | Main baseline |
| Detecting Audio Deepfakes on the Edge | arxiv.org/abs/2606.30780 | Current edge SOTA, our efficiency reference |
| ASVspoof 2021 evaluation plan | asvspoof.org | LA vs PA vs DF distinction, protocol conventions |
| Indic-CodecFake | arxiv.org/abs/2604.19949 | Indic-language prior art, cite to differentiate |

Living bibliography: github.com/john852517791/awesome-fake-audio-detection

### 3.2 Datasets

Scope: **English only.** The acoustic channel is language-independent, so
multilingual training would confound language with channel. Multilingual
validation is future work, stated explicitly in limitations.

Matching principle: ReplayDF is derived from M-AILABS (bona fide) and MLAAD
(fake). We train on those same two corpora so the only variable separating
training from test is the acoustic channel.

**Train**

| Dataset | Take | Size | Link |
|---|---|---|---|
| M-AILABS | `en_US` + `en_UK` | 11 GiB | ics.tau-ceti.space/data/Training/stt_tts/en_US.tgz (and en_UK.tgz) |
| MLAAD | English only, capped per TTS model | ~2 GB | huggingface.co/datasets/mueller91/MLAAD |
| ASVspoof 2019 | `LA.zip` only, for metric validation | 7.1 GiB | datashare.ed.ac.uk/handle/10283/3336 |

Skip ASVspoof `PA.zip` (16.4 GiB, wrong track: PA replays real human speech).

**Test only, never train on these**

| Dataset | Size | Link |
|---|---|---|
| ReplayDF | ~15 GB | deepfake-total.com/replay_df |
| EchoFake | 3 GB | github.com/EchoFake/EchoFake |

Both are gated non-commercial releases. Request early; approval takes 1-2 weeks.

**Simulator ingredients**

| Resource | Size | Link |
|---|---|---|
| MUSAN (noise) | 11 GB | openslr.org/resources/17/musan.tar.gz |
| OpenSLR 28 (RIRS_NOISES: RWCP + REVERB + Aachen AIR, real + simulated) | ~4 GB | openslr.org/resources/28/rirs_noises.zip |
| OpenSLR 26 (simulated, 16 kHz) | 178 MB | openslr.org/26 |
| MIT IR Survey (271 real, 16 kHz) | 15 MB | huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses |
| BUT ReverbDB (RIRs only) | ~300 MB | fit.vut.cz/research/group/speech |
| More RIR sets | - | github.com/RoyJames/room-impulse-responses |
| pyroomacoustics (synthetic) | pip | github.com/LCAV/pyroomacoustics |

Filter RIRs on load: drop T60 > 2.0s or DRR outside -10 to 30 dB.
Tag every RIR with its source corpus so a whole set can be held out as an
unseen-room-distribution ablation.

Total: roughly 55 GB.

**Not downloading:** Common Voice, AI4Bharat, IndicSynth, WaveFake,
In-the-Wild, DFADD, HAV-DF, Indic-CodecFake, ASVspoof 2021.
IndicSynth returns only if the optional cross-lingual section happens.

### 3.3 Compute and hardware

- RTX A5000, 24 GB VRAM, ~200 GB free, native Windows 11 (no WSL). Not shared, but Windows Update reboots are a real pre-emption risk, so every run checkpoints per epoch and resumes from `--resume`.
- Deployment measurement: borrow a Raspberry Pi from the department IoT lab. Fallback is an Android phone with the TFLite benchmark tool.
- Optional but recommended, roughly Rs 1500: one Bluetooth speaker plus an INMP441 mic. Not for the training corpus. Used to record a 200-clip "unseen device" set that appears in neither ReplayDF, EchoFake, nor the simulator. One extra results column, disproportionate reviewer value.

---

## 4. The simulator

The core of the project. Models the physical chain a replay attack must pass through.

```
clean audio
  -> loudspeaker frequency response   (band-limit, HF rolloff, resonances)
  -> loudspeaker nonlinearity         (soft clip + memoryless polynomial)
  -> room impulse response            (measured RIR bank + pyroomacoustics synthesis)
  -> background noise                 (MUSAN, sampled SNR)
  -> microphone response              (filter + self-noise)
  -> codec                            (Opus low-bitrate, AMR-NB, MP3)
  -> simulated far-field capture
```

**Domain randomisation is the whole trick.** Every parameter is resampled per training example, with ranges deliberately wider than physical reality. If every simulated room is 4x4m with the same absorption, the model overfits the simulator and fails on ReplayDF.

Randomised ranges: room dimensions, absorption coefficients, source-mic distance 0.2m to 4m, speaker cutoff frequency, distortion order and strength, SNR, codec and bitrate, mic self-noise floor.

Implemented as a GPU dataloader transform, not offline preprocessing. Convolution is cheap on GPU, so every epoch sees a fresh channel realisation. Effectively infinite augmentation.

**Validate before training.** Compute spectral tilt, C50, RT60 and HF energy distributions for simulated audio and for ReplayDF. Overlay the histograms. If the distributions do not overlap, fix the simulator before spending GPU hours. This plot goes in the paper.

---

## 5. The detector

Not looking for synthesis artifacts. Looking for evidence the sound came out of a loudspeaker.

**Features**

| Feature | Cue it captures |
|---|---|
| HF rolloff / spectral tilt | Speakers die above 8-15kHz, vocal tracts do not |
| C50, blind RT60 | Replay stacks two rooms, giving double reverb |
| LP residual kurtosis, HNR | Excitation structure disturbed by playback chain |
| Harmonic distortion ratio | Nonlinearity no vocal tract produces |
| Modulation spectrum (4-16Hz) | Double reverb smears speech modulation |

**Models**

1. LightGBM over the features. Trains in minutes, gives the feature-importance figure, is the interpretability story.
2. Small CNN over the same features, roughly 100k parameters. The deployable version.

**Formulation:** three-class (live human / replayed human / replayed synthetic), not binary.

**Fusion option:** late-fuse the channel score with a distilled artifact detector. Report both alone and fused. Reviewers like knowing which half does the work.

---

## 6. Experiments

### Table 1: sim-to-real transfer (headline)

Train on IndicSynth + MLAAD + ASVspoof, evaluate on real replay data never seen in training.

| Training condition | ReplayDF EER | EchoFake open-set EER |
|---|---|---|
| Clean audio only | (expect collapse) | (expect collapse) |
| Standard aug (noise + RIR only) | | |
| **Full physics simulation (ours)** | | |
| Trained on real replay data (oracle upper bound) | | |

Success: row 3 lands close to row 4.

### Table 2: does compression compound the damage

| Model | Clean EER | Replayed EER | Degradation |
|---|---|---|---|
| AASIST FP32 | | | |
| AASIST distilled | | | |
| AASIST INT8 | | | |
| Ours FP32 | | | |
| Ours INT8 | | | |

### Table 3: unseen generator generalisation

Hold out two TTS systems entirely. Report the seen-to-unseen gap for each model. Our gap should be smaller because channel cues are generator-independent.

### Table 4: efficiency

Params, MACs, latency, peak RAM, **millijoules per inference on real hardware**. Most papers stop at params. Measuring energy is a differentiator.

### Ablations

Leave out each simulator stage (nonlinearity, RIR, codec, mic response) one at a time and report the ReplayDF EER hit. Tells the reader which physics actually matters.

### Breakdowns

Results split by distance, room, language, and TTS system. Shows the effect is acoustic rather than incidental noise.

---

## 7. Milestones

| Week | Deliverable | Done when |
|---|---|---|
| 1 | Pilot + dataset access requested + 6 papers read | `results/week1_pilot.md` shows the EER jump |
| 2-3 | Eval harness | Reproduce AASIST's published ASVspoof19-LA EER within 0.5 points |
| 4-5 | Simulator v1 | `src/sim/channel.py` runs end to end, output sounds right by ear |
| 6-7 | Simulator validation | Statistics overlay against ReplayDF is in `results/` |
| 8-9 | Main training runs | Table 1 filled |
| 10-11 | Ablations + unseen generator | Tables 2 and 3 filled |
| 12-13 | Distil, quantise, deploy | Table 4 filled with real energy numbers |
| 14 | Own-rig recording (optional) | 200 clips, extra results column |
| 15-16 | Write, release code and simulator | arXiv preprint up |

About four months at a steady pace. Overlaps with the SASV final-year project, since the eval harness and baselines are shared.

---

## 8. Repository layout

```
airgap-add/
  PLAN.md                    this file
  README.md
  requirements.txt
  configs/                   yaml experiment configs, one per table row
  src/
    sim/
      speaker_response.py    loudspeaker filtering
      nonlinear.py           soft clipping, polynomial distortion
      rir.py                 measured RIR bank + pyroomacoustics
      codec.py               opus / amr / mp3 round-trip
      channel.py             full chain, the core file
    data/
      loaders.py             unified interface over all datasets
      augment_gpu.py         simulator as a GPU dataloader transform
      protocols.py           train/test/held-out splits
    features/
      spectral.py            tilt, HF rolloff
      reverb.py              C50, blind RT60
      residual.py            LP residual, HNR, distortion
      extract.py             orchestration + disk cache
    models/
      channel_net.py         small CNN
      gbm.py                 LightGBM
      baselines.py           AASIST, RawNet2 wrappers
      train.py               resumable training loop
    eval/
      metrics.py             EER, min-DCF
      run_eval.py            one command, any model, any test set
    deploy/
      export_onnx.py
      quantize.py
      bench.py               latency, RAM, energy
  scripts/                   download, preprocess, reproduce-table-N
  notebooks/                 exploration only, nothing load-bearing
  data/                      gitignored
  results/                   committed, this is the paper's evidence
  paper/                     latex
```

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Pilot fails, replay does not degrade detectors on our setup | Week 1 kill criterion. Cost is 3 days, not 3 months. |
| Simulation does not transfer to real replay | Validate distributions before training. If transfer is poor, the negative result is still publishable: "simulation is not enough, here is the gap." Pivot to a hybrid using a small amount of real data. |
| Reviewer: "this is just ASVspoof PA" | Cite PA in intro paragraph two, differentiate in paragraph three. PA replays real human speech; we handle replayed synthetic speech as a three-class problem. |
| Reviewer: "your simulator was tuned to ReplayDF" | Hold out EchoFake open-set and the own-rig recordings entirely. Never tune simulator parameters against them, and say so explicitly. |
| Shared GPU pre-empts runs | Per-epoch checkpointing, file logging, `--resume` on every script. |
| Dataset access delayed | Request week 1. Prototype the simulator on Common Voice while waiting. |
| No IoT hardware | Say "far-field" and "edge" in the title, not "IoT". Motivate with IoT, do not claim a device you do not have. |

---

## 10. Paper skeleton

1. **Intro.** Detectors are benchmarked on audio that physical devices never receive. Replay laundering is the mechanism. Recording real replay data does not scale.
2. **Related work.** ReplayDF and EchoFake (problem established, fix is data collection). Edge detection literature (evaluated on clean audio). ASVspoof PA (different threat model, differentiate here).
3. **Threat model.** Three classes. Attacker plays synthetic speech near a far-field microphone.
4. **Method.** The simulation chain, domain randomisation, channel features, the small model.
5. **Experiments.** Tables 1-4 plus ablations.
6. **Discussion.** When simulation is sufficient and when it is not. Limitations, honestly stated.
7. **Release.** Code, simulator, own-rig recordings.

---

## 11. Open decisions

- Is the fused channel-plus-artifact system the headline, or is the pure channel detector cleaner as a story? Decide after Table 1.
- Is the calibration chirp idea (device measures its own mic response at install, conditions the detector on it) in scope, or a follow-up paper? Cut if behind schedule; the paper stands without it.
- Which two TTS systems to hold out. Pick after inspecting IndicSynth's generator list.
