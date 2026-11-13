# Replay-Robust Speech Deepfake Detection — Project Documentation

Complete account of what this project is, what was built, what was found,
and what did not work. Written to be readable by someone who has not seen
the code.

---

## 1. The problem

Speech deepfake detectors work well on clean audio files. They break when
the audio is **replayed** — played through a loudspeaker and re-recorded
by a microphone. Room echo, speaker distortion and microphone response
smear the subtle artefacts detectors rely on.

Published evidence (EchoFake, arXiv 2510.19414): W2V2-AASIST goes from
4.7% to 18.2% EER under replay. Detectors trained on clean ASVspoof19-LA
and evaluated on replayed audio land at 42.9-46.3% EER — barely better
than a coin flip.

The field's answer is to record more real replay data. That does not
scale: every new microphone, room and distance needs fresh recordings.

**This matters for IoT specifically.** An attacker cannot inject a digital
file into a smart speaker — they must play it through the air into the
device's microphone. Replay is not an edge case for voice assistants, it
is the entire attack surface.

---

## 2. The hypothesis

> The acoustic replay channel (loudspeaker → room → microphone) can be
> *simulated* well enough that a detector trained only on simulated data
> generalises to real re-recorded audio.

If true, replay robustness stops requiring physical data collection.

**Success criterion, set in advance** (CLAUDE.md section 2): below ~43%
EER on EchoFake open-set supports the hypothesis; ~15% would be a strong
paper; at or above 43% means simulation did not transfer.

---

## 3. What was built

### 3.1 The replay channel simulator (`src/sim/`)

Four stages, chained in the order sound physically travels:

| Stage | File | What it models |
|---|---|---|
| Speaker response | `speaker_response.py` | Loudspeaker EQ colouring: bass roll-off, cabinet resonances, HF roll-off. Randomised per call. |
| Nonlinear distortion | `nonlinear.py` | Amplifier/driver saturation. Generates real harmonics via `tanh` waveshaping — something no linear filter can do. |
| Room impulse response | `rir.py` | Convolution with a real measured RIR. Produces echo and reverb tail by smearing the signal in time. |
| Codec | `codec.py` | Lossy compression round-trip through real ffmpeg encoders (MP3, Opus, AMR-NB, G.711 µ-law). |
| Chain | `channel.py` | Composes all four with a single seeded RNG. |

Every stage was validated individually before chaining:

- **Speaker response**: power-spectral-density plots confirm sensible,
  seed-varying filter shapes.
- **Nonlinear**: a 440 Hz pure tone produces new energy at exactly
  1320 Hz and 2200 Hz (3rd and 5th harmonics) and *zero* at even
  harmonics — the mathematically correct signature of a symmetric
  nonlinearity.
- **RIR**: spectrograms show monotonically increasing time-smearing as
  room size grows (0.5s → 0.99s → 1.99s RIRs).
- **Codec**: AMR-NB and G.711 collapse to the noise floor at exactly
  3.4 kHz, the telephony bandwidth limit; MP3 and Opus track full band.

### 3.2 Data pipeline (`src/data/`)

- `loaders.py` — every corpus normalised to 16 kHz mono with silence
  trimmed by one uniform threshold. Both steps mandatory, not optional
  flags: M-AILABS ships a fixed 0.5s pad that would otherwise be a
  "dead air ⇒ genuine" shortcut.
- `protocols.py` — train/dev manifests. Split unit is the **book**, and
  MLAAD fakes inherit the split of the source recording they were
  synthesised from, so the same sentence never appears as genuine in
  train and as a fake in dev.
- `augment_gpu.py` — PyTorch dataloader applying a fresh random channel
  per sample per epoch, seeded from `(run_seed, epoch, index)` so runs
  reproduce exactly across multiple worker processes.
- `generate_clones.py` / `generate_asv_clones.py` /
  `generate_echofake_clones.py` — XTTS-v2 voice cloning, resume-safe.

### 3.3 Features (`src/features/`)

- `spectral.py` — MFCCs and spectral shape statistics.
- `reverb.py` — blind reverberation indicators (spectral tilt, HF energy,
  modulation depth — the last based on the Speech Transmission Index
  principle that reverb flattens syllabic-rate modulation).
- `residual.py` — LP residual. Linear prediction models the vocal tract;
  subtracting its prediction leaves the excitation signal, where channel
  and synthesis artefacts concentrate.
- `w2v2.py` — frozen Wav2Vec2 embeddings, mean+std pooled over a
  **fixed-length** window (see §6.2 for why fixed length matters).
- `enrollment.py` — features of a test utterance *relative to* an
  enrolled speaker's centroid, for the per-user setting.

### 3.4 Evaluation (`src/eval/`)

- `metrics.py` — EER via Schroeder-style DET curve construction.
- `leakage_check.py` — asserts no held-out path appears in a training
  manifest. Verified to actually catch an injected leak, not just pass.
- `run_eval_echofake*.py`, `run_eval_iot_enrolled.py` — evaluation
  harnesses for each model type.

---

## 4. Verification gates

The project used pre-declared gates; each had to pass before the next
stage. Results:

| Gate | Requirement | Outcome |
|---|---|---|
| G0 Pilot | Replay degrades detection | **PASS** via published evidence (no recording hardware available; deviation documented) |
| G1 Metrics | Reproduce AASIST's published number within 0.5 pts | **PASS, exact** — 0.83% vs 0.83% published |
| G2 Loaders | Assertion suite on splits, rates, silence | **PASS** |
| G3 Simulator | Acoustic distributions overlap ReplayDF | **PASS on 3 of 4** (T60 not validatable, see §6.4) |
| G4 Leakage | No held-out path in training | **PASS**, and verified to catch injected leaks |
| G5 Baseline | Clean baseline near published ~43% | **FAIL** — landed at or worse than chance |

G1 passing *exactly* is the load-bearing one: it means every EER reported
here is measured correctly.

---

## 5. Results

All EER; lower is better; 50% is chance.

### 5.1 Speaker-independent detection (the original hypothesis)

| Model | In-domain | EchoFake open-set |
|---|---|---|
| Hand-crafted features, clean | 0.50% | 56.50% |
| Hand-crafted features, replay-augmented | 25.80% | 48.27% |
| Same-speaker clones, clean | 0.27% | 50.92% |
| Same-speaker clones, replay-augmented | 13.79% | **46.71%** |
| Frozen W2V2, replay-augmented | 15.34% | 47.66% |
| *Published: RawNet2 / AASIST / W2V2* | — | *46.34 / 43.23 / 42.94* |
| *Published oracle (trained on real replay)* | — | *11.86* |

**Verdict: negative.** 46.71% is above the 43% bar set in advance.
Training only on simulated replay did not transfer well enough.

**What did hold:** replay augmentation consistently helps. On identical
features, model and data, it moved 50.92% → 46.71% (open-set) and
47.54% → 44.83% (closed-set). Real, reproducible, but not sufficient.

### 5.2 Per-speaker detection (the IoT framing)

A device is enrolled to one person. The task becomes "is this my enrolled
user, or a clone of them?" rather than "is this synthetic in general".

| Setup | EER |
|---|---|
| Per-speaker detectors, 4 speakers, content-matched | **12.25%** |
| Same, with resampling control | **12.00%** |

Per speaker: elliot_miller 11.67%, judy_bieber 12.00%, mary_ann 12.00%,
elizabeth_klett 13.33% — consistent.

### 5.3 Cross-generator test — the deciding experiment

Per-speaker detectors trained on XTTS-v2 clones of 107 EchoFake speakers,
evaluated against fakes of the same speakers by five unseen generators:

| Condition | Trials | EER |
|---|---|---|
| Held-out clones, same generator (positive control) | 480 | **22.28%** |
| Five unseen generators | 4,560 | **53.14%** |

Per generator: OpenVoice-V2 48.30%, FireRedTTS 52.36%, IndexTTS 52.67%,
CosyVoice2 55.11%, MaskGCT 59.50%.

**Negative, and well-controlled.** The positive control shows the pipeline
works, so 53.14% is real. A personal detector learns the fingerprint of the
generator it trained on, not synthetic speech in general. The 12.00% above
is same-generator performance only and is not evidence of deployable
protection. See `results/per_speaker_cross_generator.md`.

---

## 6. Confounds found

Four times a result looked good; each time investigation showed the gain
was an artefact. This section is the most transferable part of the work.

### 6.1 Voice-identity confound in MLAAD + M-AILABS

MLAAD's README **explicitly recommends** pairing its fakes with M-AILABS
genuine audio. But MLAAD's fakes are not clones of the M-AILABS speakers:

| Audio | Median F0 |
|---|---|
| Genuine `elliot_miller` | 99.5 Hz |
| MLAAD `xtts_v2` fake *from his own text* | 205.3 Hz |
| MLAAD `OpenVoiceV2` fake from his text | 169.6 Hz |

MLAAD ran cloning-capable models in *default-voice* mode. At least 18 of
its 36 English models use a fixed foreign voice (9 are LJSpeech).

So in that recommended pairing, genuine is 4 voices and fake is a
collection of *other* voices — the task is partly solvable by speaker
identification. Replacing MLAAD fakes with same-speaker clones improved
cross-corpus transfer by ~9 points, **and shrank the apparent
augmentation benefit from 8.2 to 2.7 points**, showing part of the
original gap was the confound rather than the simulator.

ReplayDF is built on this same pairing.

*Caveat, stated honestly:* pitch alone separates the classes at only
43.33% EER, so the crude version of this shortcut is disproven. Whether
models exploit richer timbre identity is plausible but unproven.

### 6.2 Duration cue

An apparent breakthrough — 5.86% EER on unseen speakers, better than the
published oracle — turned out to be length. ASVspoof genuine clips have
median duration 1.76s; our clones spoke M-AILABS book sentences at median
7.44s. **Duration alone classified at 6.00% EER.** W2V2 mean+std pooling
over variable-length audio encodes length directly.

Fixed-length cropping (2.5s) removed it: the honest number was 11.25%.

### 6.3 Content mismatch

The 11.25% did not survive cross-corpus evaluation (47.55% on EchoFake).
The likely cause is content: our genuine audio is VCTK newspaper prompts
while our clones read Victorian novels, whereas EchoFake's genuine and
fake share one content pool. A detector keyed on "prompt speech vs
literary speech" scores exactly like that — excellent locally, chance
elsewhere.

### 6.4 A measurement artefact in ReplayDF's published RIRs

ReplayDF's `aux/*/RIR.wav` files show a near-identical, razor-sharp
cutoff at exactly 4000 Hz across all 110 recordings, regardless of which
of its 109 speaker/mic combinations produced them — including a Blue Yeti
condenser, which has no business producing that. It is a digital artefact
of their sweep-derived RIR extraction, not the replay channel.

Consequence: a brick-wall filter's sinc ringing in the time domain is
misread by Schroeder integration as reverberation. **67% of ReplayDF's
published RIRs yield T60 estimates above 5 seconds**, up to 116 seconds —
physically impossible for ordinary rooms with JBL/Bose speakers on desks.

The artefact is confined to those diagnostic files: ReplayDF's actual
replayed *speech* shows normal spectra. G3's tilt/HF metrics were
therefore re-derived from matched dry/replayed speech pairs instead.

### 6.5 A methodological near-miss worth recording

The first resampling control returned *exactly* 5.86% — identical to the
uncontrolled run to two decimal places. That identity is what exposed it:
the control had monkeypatched a function in the parent process, but
Windows **spawns** worker processes which re-import modules fresh, so the
patch never reached them and the "control" recomputed identical features.
Controls must travel to workers as data, not as patched globals.

---

## 7. Honest limitations

- **Four bona fide speakers** in the primary corpus. English M-AILABS
  contains four voices, and rule 3 tied us to it for ReplayDF lineage.
  This is the most likely ceiling on speaker-independent generalisation.
- **One synthesis model** (XTTS-v2) in clone training, against
  EchoFake's six.
- **Small models** — LightGBM on summary statistics, a 187K-parameter
  CNN, and frozen (not fine-tuned) W2V2. The published references are
  far larger and fine-tuned end-to-end.
- **T60 unvalidated** against ReplayDF, per §6.4.
- **Enrolment uses held-out audio** in the IoT experiments — standard in
  speaker verification, but a departure from this project's rule 1 and
  disclosed wherever reported.

---

## 8. Reproducing

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt

# gates
PYTHONPATH=. python tests/test_metrics.py         # G1 unit checks
PYTHONPATH=. python src/eval/run_eval.py          # G1 against AASIST
PYTHONPATH=. python tests/test_protocols.py       # G2
PYTHONPATH=. python tests/test_loaders.py         # G2

# simulator demos (writes audio + plots to results/sim_demos/)
PYTHONPATH=. python scripts/bench_dataloader.py

# experiments
PYTHONPATH=. python scripts/train_clean_lgbm.py
PYTHONPATH=. python scripts/run_same_speaker_experiment.py
PYTHONPATH=. python scripts/train_per_speaker.py
```

Data is not committed (rule 6). `data/raw/README.md` documents layout and
per-corpus gotchas.

**Windows note:** anything using `DataLoader` or `ProcessPoolExecutor`
must be a real script file with an `if __name__ == "__main__":` guard —
spawned workers re-import the main module, and a stdin heredoc gives them
no importable path.

---

## 9. Current state

- Speaker-independent hypothesis: **evaluated, negative** (46.71% vs 43%
  bar).
- Per-speaker IoT framing: **12.00% same-generator, 53.14% against five
  unseen generators** (positive control 22.28% confirms the pipeline).
  Personalisation alone does not generalise across generators.
- Multi-generator training tested and **does not help**: 52.95% with two
  architecturally distinct generators vs 53.14% with one. The obvious
  remedy fails, which closes this line rather than leaving it open.
- Simulator, gates, evaluation harness: complete and validated.
