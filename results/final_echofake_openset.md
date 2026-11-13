# Final evaluation: EchoFake open-set

One-shot evaluation, run once after the pipeline was validated on
closed-set only (rule 2: open-set never inspected before final eval).

## Result

| Model | EchoFake open-set EER |
|---|---|
| Same-speaker, clean-trained | 50.92% |
| **Same-speaker, simulated-replay-trained** | **46.71%** |

Published reference points (CLAUDE.md section 2, trained on ASVspoof19-LA
clean, evaluated on EchoFake open-set):

| Model | EER |
|---|---|
| RawNet2 | 46.34% |
| AASIST | 43.23% |
| Wav2Vec2 | 42.94% |
| EchoFake-train oracle (real replay, Wav2Vec2) | 11.86% |

## Verdict against the project's own criteria

CLAUDE.md section 2 set the bar: *"Below ~43% EER on EchoFake-open: the
hypothesis has support. Near ~15%: strong paper. At or above 43%:
simulation did not transfer."*

**46.71% is above 43%. By the project's own stated criterion, this is a
negative result on the primary hypothesis:** training only on simulated
replay did not transfer well enough to match a detector trained on real
replay data, nor to clear the threshold set in advance.

Reported as a finding, not a failure, per CLAUDE.md's instruction.

## What the experiment does establish

**1. The simulated replay channel measurably helps.**

| Comparison | Clean | Augmented | Gain |
|---|---|---|---|
| Open-set | 50.92% | 46.71% | 4.21 pts |
| Closed-set | 47.54% | 44.83% | 2.71 pts |

Identical features, model class, hyperparameters and data — the only
difference is whether the simulated channel was applied. The clean model
sits at literal chance on open-set (50.92%); augmentation moves it
meaningfully off chance. The simulator does something real, it just
isn't sufficient on its own.

**2. A voice-identity confound in the training data was found and fixed.**

MLAAD's fakes are not clones of the M-AILABS speakers (genuine
`elliot_miller` 99.5 Hz vs MLAAD's "fake from his text" at 205.3 Hz), so
the original task was partly solvable by speaker identification. Replacing
MLAAD fakes with XTTS-v2 clones of the same four speakers improved
cross-corpus transfer substantially:

| Training data | Clean | Augmented |
|---|---|---|
| MLAAD fakes (confounded) | 56.50% | 48.27% |
| Same-speaker clones | 47.54% | 44.83% |

Note this *reduced* the apparent augmentation benefit (8.23 pts -> 2.71
pts on closed-set), i.e. part of the original gap was an artifact of the
confound. Verified the fix introduced no replacement shortcut: pitch
alone separates genuine from clone at 49.38% EER (chance), versus 43.33%
on the MLAAD-based data.

**3. Competitive efficiency, in context.** 46.71% from 58 hand-crafted
features and LightGBM is close to RawNet2's 46.34%, a deep architecture
trained on ASVspoof19-LA (~100 speakers, 20+ attack types) versus our 4
speakers and a single TTS system. Not a win; worth noting for the
efficiency argument.

## Honest limitations

- **Four bona fide speakers.** English M-AILABS contains four voices.
  This is the most likely ceiling on learning speaker-independent,
  corpus-independent synthesis-artifact detection, and it is baked into
  the rule-3 requirement to match ReplayDF's lineage.
- **One synthesis model in training** (XTTS-v2) against EchoFake's six
  (StyleTTS2, LLaSA, XTTS-v2, OpenAudio-S1-mini, F5-TTS, SpeechT5).
  ~17% overlap, so mostly a cross-model test, but not a clean one.
- **Small models.** LightGBM on summary statistics, and a 187K-parameter
  CNN. The published references are far larger; Wav2Vec2 additionally
  benefits from large-scale self-supervised pretraining, which is the
  standard remedy for exactly this data-scarcity regime.
- **T60 not validated** against ReplayDF (see
  `g3_simulator_validation.md`) due to a measurement artifact in
  ReplayDF's published RIR files.

## Suggested next step

The most likely path to clearing the 43% bar is a pretrained backbone
(Wav2Vec2/WavLM-style) rather than from-scratch models, since the
evidence points to training-data diversity as the binding constraint, not
the simulator. The simulator's contribution is already demonstrated and
would carry over.

Date: 2026-09-08
