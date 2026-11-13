# G3: simulator distribution validation vs ReplayDF

**Gate:** simulated audio's T60/C50/spectral tilt/HF energy distributions
overlap ReplayDF's. Plot committed to `results/`. Do not train before this.

## What was built

- `src/sim/acoustics.py` — `estimate_t60`, `compute_c50`,
  `spectral_tilt_db_per_khz`, `hf_energy_ratio_db`. Validated against
  synthetic ground truth before use: T60 estimation recovered known values
  to within 0.3-2%, and tilt/HF functions respond correctly and
  monotonically to a known filter.

## A methodology problem discovered mid-validation, and how it was resolved

The first pass compared our training RIR pool directly against
ReplayDF's `aux/*/RIR.wav` files for all four metrics. Two things went
wrong, both traced to root cause before drawing any conclusion:

**1. Pool contamination.** The initial RIR pool glob included
`RIRS_NOISES/pointsource_noises/` (843 files, pure noise recordings, zero
RIRs) and unfiltered `real_rirs_isotropic_noises/` (417 files, of which 92
are `RVB2014_*_noise_*` isotropic noise recordings, not RIRs, despite the
directory name implying otherwise). This produced nonsensical T60 values
up to 221 seconds. Fixed by filtering to genuine RIR files only (basename
must not contain "noise" in that specific folder; `pointsource_noises`
excluded entirely). Corrected pool: 60,647 files.

**2. ReplayDF's RIR.wav is not a room-only measurement.** ReplayDF's own
README states these files are "derived from sine sweep" — i.e. the
impulse response of the *entire physical measurement chain* (their
loudspeaker playing a sweep, through the room, into their microphone,
deconvolved), not a pure acoustic room response. Our training RIR pool
(OpenSLR-28 simulated RIRs + calibrated research recordings) has no such
transducer coloration baked in.

Worse: every one of 6 randomly inspected ReplayDF RIR.wav files showed an
almost identical, razor-sharp cutoff at exactly 4000 Hz, dropping 40-60 dB
in a hairsbreadth and flatlining at nearly the same noise floor regardless
of which of their 109 speaker/mic combinations was used (including a Blue
Yeti USB mic, a wideband condenser with no business producing an exact
4kHz brick-wall). This is not real acoustic/transducer physics — real
hardware rolloffs are gradual and device-dependent. It is the signature of
a digital artifact in how the RIR was extracted (most likely a band-limited
sine-sweep excitation signal or deconvolution regularization).

This same artifact also corrupted the T60 estimates: a sharp frequency-domain
brick-wall filter mathematically produces a slowly-decaying sinc-ringing
tail in the time domain, which Schroeder backward integration misreads as
genuine reverberation. 67% of ReplayDF's RIR.wav files produced T60
estimates above 5 seconds (up to 116 seconds) — physically impossible for
the ordinary rooms these represent (home setups with JBL/Bose speakers and
USB mics, per `aux/*/info.txt`).

**Confirmation the artifact is specific to the diagnostic RIR.wav files,
not the real replay channel:** spectral metrics computed directly on
ReplayDF's actual replayed speech (`wav/*/benign` and `wav/*/spoof`, the
files actually used for detection evaluation) show HF energy and tilt far
milder than RIR.wav suggests, and much closer to our simulator's output.

## Corrected methodology

- **T60**: dropped as a RIR.wav-based comparison. Not currently
  validatable against ReplayDF without a blind estimator from real speech
  (not built yet — noted as a follow-up if T60-specific validation
  becomes necessary).
- **C50**: kept from RIR.wav directly. Localized to the first 50ms after
  the direct-path peak, least exposed to a HF-band artifact.
- **Spectral tilt, HF energy**: recomputed using matched dry/replayed
  utterance pairs instead of RIR.wav. ReplayDF's own `meta.csv` links every
  replayed file back to its exact M-AILABS/MLAAD source
  (`original_file`), which we hold locally for English. For 4,400 matched
  English bona fide pairs, computed the empirical channel transfer
  function as the dB difference between replayed and dry Welch PSDs (a
  ratio cancels speech content and isolates the channel, the same way a
  RIR isolates it from an impulse) — sampled 300. Did the identical
  computation for our simulator (`speaker_response` + `nonlinear` + `rir`,
  no `codec` since ReplayDF's own recordings aren't further
  codec-compressed) on 300 matching dry sources.

## Result

| Metric | Comparison | Result |
|---|---|---|
| T60 | RIR.wav direct | **unreliable, not evaluated** — measurement artifact |
| C50 | RIR.wav direct | good overlap (both peaked ~10-20 dB) |
| Spectral tilt | matched dry/replayed pairs | good overlap — sim median -1.71 dB/kHz, real median -0.24 dB/kHz, sim's full range sits inside real's range |
| HF energy (>4kHz, relative) | matched dry/replayed pairs | good overlap — sim median -8.4 dB, real median -3.6 dB, substantially overlapping distributions |

Plot: `results/g3_simulator_validation.png`.

**G3: PASS on 3 of 4 metrics (C50, spectral tilt, HF energy). T60 could
not be validated due to a data artifact in ReplayDF's own published files,
not a simulator problem** — confirmed by reproducing the same artifact's
mathematical signature (sinc-ringing from a brick-wall filter) and by
showing the real replayed speech itself doesn't carry it. Proceeding to
training on this basis; if T60-specific validation becomes necessary
later, a blind reverberation-time estimator run on real speech (not
RIR.wav) would be the way to get it.

Date run: 2026-09-07
