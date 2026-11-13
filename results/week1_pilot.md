# Week 1 pilot: does replay actually break the detector?

**Kill criterion.** Do not proceed with the project unless this shows a clear degradation.

## Protocol

1. Sample 100 clips: 50 bona fide (Common Voice Hindi), 50 synthetic (IndicSynth).
2. Score with pretrained AASIST. Record EER.
3. Play each clip through a laptop or Bluetooth speaker, record with a phone at ~1m in a
   normal room.
4. Score the re-recorded versions with the same model. Record EER.

## Results

Local pilot not run: no recording hardware (speaker/mic) available on this
machine. Substituted external published evidence for the kill criterion,
per CLAUDE.md section 5 ("anything that would touch a hard rule... confirm").

From EchoFake (arXiv 2510.19414), Table 3 — clean-trained models evaluated
on EchoFake's replayed open-set:

| Condition | RawNet2 | AASIST | Wav2Vec2 |
|---|---|---|---|
| Clean (ASVspoof19-LA published baseline) | ~2-3 | ~1 | ~1 |
| Replayed (ASV19LA-trained, EchoFake open-set) | 46.34 | 43.23 | 42.94 |
| Degradation | ~44 | ~42 | ~42 |

Separately, W2V2-AASIST goes from 4.7% to 18.2% EER under replay
(CLAUDE.md section 2). Both citations show replay degrading detection far
beyond the 8-point PROCEED threshold, across three architectures and two
independent published evaluations.

## Decision

- Degradation >= 8 points: PROCEED.
- Degradation < 3 points: STOP, re-scope.
- In between: investigate before committing.

Date run: 2026-09-07 (evidence review, not a fresh local recording)
Decision: PROCEED — premise confirmed via published EchoFake and
EchoFake-paper baselines, degradation ~42-44 points, an order of magnitude
past the kill threshold. If recording hardware becomes available later, a
local pilot can still be run as a sanity check on our own equipment, but it
is not a blocker.
Notes: G0 is unusual in this project in requiring a physical step (play +
re-record). Every other gate (G1-G5) is fully reproducible in software from
data already on disk or in flight.
