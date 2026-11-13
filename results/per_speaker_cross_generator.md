# Per-speaker detection: does it survive an unseen generator?

The deciding experiment for the IoT framing. A device enrols its owner,
synthesises clones of them locally, and trains a personal detector. The
attacker then uses whatever TTS they like — not the one the device used.

## Setup

For each of 107 EchoFake speakers:

- **Train:** that speaker's genuine audio vs 40 XTTS-v2 clones of them
  that we generated. Simulated replay applied to ~50% of clones, matching
  EchoFake's own proportion of replayed genuine audio (without this the
  model could learn "replayed ⇒ genuine").
- **Test:** held-out genuine vs EchoFake's fakes *of that same speaker*.

The detector never sees an EchoFake fake during training, so every
generator at test time is genuinely unseen.

EchoFake's open split contains **no XTTS-v2 fakes at all** — its generator
roster differs entirely from its closed split. That is a good property of
the corpus and means all five test generators are unseen.

## Result

| Condition | Trials | EER |
|---|---|---|
| Held-out clones, same generator as training (positive control) | 480 | **22.28%** |
| EchoFake fakes, 5 unseen generators | 4,560 | **53.14%** |

Per unseen generator:

| Generator | n | EER |
|---|---|---|
| OpenVoice-V2 | 826 | 48.30% |
| FireRedTTS | 735 | 52.36% |
| IndexTTS | 816 | 52.67% |
| CosyVoice2 | 768 | 55.11% |
| MaskGCT | 773 | 59.50% |

## Interpretation

The positive control matters: at 22.28% the pipeline demonstrably works,
so 53.14% is a real negative rather than a broken experiment. (22.28% is
weaker than the 12.00% obtained on M-AILABS speakers because training
genuine audio here is scarce — roughly 6-30 clips per speaker instead of
thousands.)

**A per-speaker detector trained on one generator's clones learns that
generator's fingerprint, not synthetic speech in general.** Against an
unseen generator it performs at or below chance. Every one of the five
sits between 48% and 60%, so this is not one awkward outlier.

This directly refutes the deployment story that motivated the experiment:
enrolling a user and synthesising clones of them locally does **not**, on
its own, confer protection against an adversary who picks a different TTS.

## Consequence for the earlier per-speaker figure

The 12.00% reported for M-AILABS per-speaker detectors trained and tested
on XTTS-v2 should be read as same-generator performance only. It is not
evidence of deployable protection.

## What would be worth trying next

Train each personal detector on clones from **several** generators rather
than one. If generator-agnostic artefacts exist, multi-generator training
is the way to reach them; if performance still collapses on a sixth unseen
generator, that would be strong evidence that per-speaker synthetic-clone
training cannot generalise at all.

Date: 2026-09-09

---

## Follow-up: does multi-generator training fix it?

The natural remedy proposed above — train each personal detector on clones
from several generators, so it must separate genuine-user from several
distinct synthetic-users rather than memorising one fingerprint.

Added YourTTS (VITS-based, architecturally distinct from XTTS's
autoregressive design) — 4,280 further clones across the same 107
speakers. Identical code, speaker splits, evaluation set, features and
classifier; the only change is training clones from two generators
instead of one.

| Unseen generator | XTTS only | XTTS + YourTTS |
|---|---|---|
| OpenVoice-V2 | 48.30% | 46.73% |
| IndexTTS | 52.67% | 49.52% |
| FireRedTTS | 52.36% | 53.74% |
| CosyVoice2 | 55.11% | 55.32% |
| MaskGCT | 59.50% | 59.22% |
| **Overall** | **53.14%** | **52.95%** |

**No effect.** Two generators improve, two worsen, and the pooled figure
moves 0.19 points — within noise. Doubling training-generator diversity
produced no directional movement, which is the outcome least compatible
with the mechanism working weakly.

### Conclusion for this line of work

Per-speaker detection trained on synthesised clones:

- works against the generator it trained on (22.28% held-out control),
- fails against unseen generators (53.14%),
- and is **not** rescued by doubling generator diversity (52.95%).

Three generators or more might behave differently, but the complete
absence of movement from one to two is weak grounds for expecting it.
The more likely reading is that a detector trained on synthetic examples
of a specific speaker learns the artefacts of the generators it saw, and
that this does not extrapolate to unseen synthesis architectures.

Date: 2026-09-09
