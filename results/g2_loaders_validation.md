# G2: loader and protocol assertion suite

**Gate:** sample rates uniform, silence handling uniform, no file appears
in more than one split, bona fide and fake come from the paired corpora.

## What was built

- `src/data/loaders.py` — `load_audio()`: every file forced to 16 kHz mono,
  silence-trimmed with one fixed threshold (`top_db=30`) regardless of
  source. Neither step is optional.
- `src/data/protocols.py` — train/dev manifest builder. Split unit is the
  M-AILABS book. Every MLAAD fake clip inherits its split from the book its
  `original_file` points to, so a sentence can never appear as bona fide in
  train and as a fake synthesis of that same sentence in dev.
- `tests/test_loaders.py`, `tests/test_protocols.py` — the assertion suite.

## Facts measured (not assumed)

- MLAAD English is uniformly 22050 Hz across all 36 TTS models (180 files
  checked, 0 outliers) — mismatched with M-AILABS's 16 kHz. Resampled down
  to 16 kHz in `load_audio`, matching the convention used by ASVspoof and
  the RIR corpora.
- M-AILABS `en_US` matches `en_UK`'s exact ~0.5s leading/trailing pad. MLAAD
  has no such convention (leading silence median 0.017s). Both are trimmed
  uniformly to remove the shortcut, not just the M-AILABS side.
- Every MLAAD `original_file` traces back into `en_UK/` or `en_US/` — rule 3
  (paired corpora) verified from the data, not assumed from folder layout.

## Split

Stratified by book, all 4 bona fide speakers (`elizabeth_klett`,
`elliot_miller`, `judy_bieber`, `mary_ann`) present in both train and dev.
`elizabeth_klett` and `mary_ann` only have 2 books each, so no whole-book
split gets them near 80/20 individually — the pooled ratio lands at:

| | bona fide | fake |
|---|---|---|
| train | 47,390 (67.8%) | 24,577 (68.3%) |
| dev | 22,463 (32.2%) | 11,423 (31.7%) |

Not the hoped-for 80/20, but consistent between bona fide and fake, and
every speaker / all 36 TTS models present in both splits. Dev here is for
early stopping and model selection only — the real generalization numbers
come from ReplayDF and EchoFake, never touched during training.

## Result

All checks pass:
- no duplicate paths within either manifest
- no path in both train and dev
- every fake clip's speaker traces to one of our 4 M-AILABS speakers
- every speaker present in both splits
- every fake clip's split matches its source bona fide recording's split
- sample rate and silence uniform after `load_audio`, both corpora

**G2 PASS.**

## Post-hoc correction (2026-09-07, during CNN training)

Two files in `en_US` turned out to be macOS AppleDouble resource-fork
junk (`._ozma_of_oz_10_f000095.wav`, `._silent_bullet_11_f000200.wav`) —
sidecar metadata files left over from a tar archive built on macOS,
matching the `.wav` glob despite not being audio. `_speaker_dirs()`
already filtered `._*` directory-level junk (e.g. `._info.txt`), but the
file-level `.wav` glob in `build_bonafide_manifest()` didn't have the
same filter. Found via a deterministic training crash
(`soundfile.LibsndfileError: Format not recognised`), not the original
G2 assertion suite — the suite's 5-file listener-style samples never
happened to draw either junk file. Fixed in `protocols.py`; bonafide
count corrected from 69,855 to 69,853. Split percentages unaffected
(67.8/32.2).

Date run: 2026-09-07
