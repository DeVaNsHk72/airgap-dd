# G5 baseline check, and a voice-identity confound in the training data

## G5 gate result: does our clean-trained baseline land near the published ~43%?

**No.** Both clean-trained baselines land *worse than chance* on EchoFake
closed-set, far from the 42-46% EER the published reference models
(RawNet2 / AASIST / Wav2Vec2, CLAUDE.md section 2) achieve when trained
clean on ASVspoof19-LA.

| Model | In-domain dev EER | EchoFake closed-set EER |
|---|---|---|
| Clean LightGBM | 0.50% | 56.50% |
| Clean CNN (187K params) | 1.25% | 54.33% |
| Augmented LightGBM (simulated replay) | 25.80% | **48.27%** |

Per CLAUDE.md's own protocol ("if it does not, our pipeline differs from
theirs and we must find out how"), this was investigated rather than
accepted or explained away.

## The one genuinely positive signal

The two LightGBM rows are a clean apples-to-apples comparison — identical
features, model class, hyperparameters and data, differing *only* in
whether the simulated replay channel was applied. Augmentation moved
cross-corpus EER from **56.50% to 48.27%**, crossing from worse- to
better-than-chance. That is real evidence that simulation-based
augmentation improves cross-corpus generalization, though still well
short of both the ~43% reference and the ~15% "strong paper" target.

## What the investigation found: bona fide and fake are different voices

MLAAD's fakes are **not clones of the M-AILABS speakers**. Verified by
pitch, which is unambiguous for a male speaker:

| Audio | Median F0 |
|---|---|
| Genuine `elliot_miller` | 99.5 Hz |
| MLAAD `xtts_v2` fake *derived from his utterances* | 205.3 Hz |
| MLAAD `OpenVoiceV2` fake from his utterances | 169.6 Hz |
| MLAAD `ljspeech_vits` | 207.7 Hz |

MLAAD synthesized his *text* in unrelated default voices — it ran
cloning-capable models in default-voice mode rather than conditioning on
the source speaker. Of MLAAD's 36 English models, at least 18 use a fixed
foreign voice (9 are LJSpeech, plus blizzard2013, jenny, sam, ek1,
LibriTTS, vctk, viVoice).

Consequence: in our training data, bona fide is 4 specific voices and
fake is a collection of *other* voices, so the task is partly solvable by
speaker identification rather than by synthesis artifacts — a shortcut
that cannot transfer to any corpus with different speakers. This is
CLAUDE.md section 9's "corpus shortcut" in a subtler form: same corpora,
but systematically different voices between classes.

**Caveat, stated honestly:** median F0 *alone* separates our classes at
only 43.33% EER, i.e. barely better than chance — three of four bona fide
speakers are female (165-197 Hz) and many MLAAD voices are also female.
So the crude pitch shortcut is ruled out. Whether the model exploits
richer voice-timbre identity (MFCCs, which dominate our feature
importances, are *the* classic speaker-recognition feature) is plausible
but **not proven**.

## EchoFake does not have this problem

EchoFake's fakes are speaker-referenced clones. In its dev split, 337
speakers appear both as bona fide *and* as the reference voice for fakes
cloned from them (top speaker: 90 bona fide, 84 cloned fakes). Its fakes
span 6 TTS systems roughly evenly (StyleTTS2, LLaSA, XTTS-v2,
OpenAudio-S1-mini, F5-TTS, SpeechT5).

So the evaluation target measures genuine synthesis-artifact detection,
while our training data may not require it — a train/test mismatch in
what the task actually *is*.

## Remedy in progress: same-speaker clones

Generating XTTS-v2 clones of the four M-AILABS speakers, conditioned on
their own audio, so both classes share voice identity. Verified working:

| | Median F0 |
|---|---|
| Reference clip (`elliot_miller`) | 111.9 Hz |
| XTTS-v2 clone of it | 112.2 Hz |

Confirmed across all four speakers (clones track their reference clip
closely; `elliot_miller` clones land at 92-113 Hz, squarely male).

Split hygiene: each clone's text and voice-reference clip come from the
same book and inherit that book's `BOOK_SPLIT` assignment, so content
never crosses train/dev — the same rule already applied to MLAAD fakes.

**Note on overlap:** XTTS-v2 is ~17% of EchoFake's fakes, so training on
XTTS-v2 clones and testing on EchoFake is mostly (83%) a cross-model
generalization test, but not entirely. To be reported, not hidden.

Date: 2026-09-07
