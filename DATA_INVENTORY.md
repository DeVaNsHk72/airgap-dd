# Data inventory and audit

Audit of everything that was dumped into the project root, done by inspecting
the bytes on disk rather than trusting download notes. Every sample rate and
count below was measured.

For the folder layout and the per-corpus gotchas that affect loader code, see
[data/raw/README.md](data/raw/README.md). This file records what arrived, what
is missing, and where the disk disagrees with `CLAUDE.md`.

Repo root: `C:\Users\admin\Documents\airgap-dd`. Free space at time of audit:
C: 229 GB, D: 949 GB.

---

## 1. What arrived, and where it now lives

| Corpus | Now at | Size | Measured format | Complete? |
|---|---|---|---|---|
| EchoFake (parquet) | `data/raw/heldout/echofake/` | 3.1 GB | audio embedded in parquet, unverified | yes |
| ReplayDF | `data/raw/heldout/replaydf/` | 5.5 GB | 22,050 Hz, 16-bit, 967 stereo files | **no, 28 of 110 UIDs** |
| ASVspoof 2019 LA | `data/raw/validation_only/asvspoof19_la/` | 7.6 GB | 16 kHz flac | yes |
| MIT IR Survey | `data/raw/rir/mit_ir_survey/` | 17 MB | **32,000 Hz, 24-bit, mono**, 270 files | yes |
| OpenSLR-28 RIRS_NOISES | `data/raw/rir/openslr28/` | 3.8 GB | 16 kHz, 16-bit | yes |
| HOMULA-RIR | `data/raw/rir/homula/` | 101 MB | **48 kHz, 32-bit float, 8ch x50 + 64ch x2** | yes |
| FLAIR | `data/raw/rir/flair/` | 116 MB | unknown, one MATLAB v5 `.mat` | contents unverified |

Source archives were moved to `data/archives/`, exact duplicates to
`data/archives/duplicates/`. Nothing was deleted.

EchoFake split sizes from its own README: train 39,926 / dev 3,973 /
closed_set_eval 5,991 / open_set_eval 25,600.

OpenSLR-28 breakdown: 843 point-source noises, 420 real-RIR and isotropic-noise
entries (RWCP + REVERB + AIR), 60,611 simulated RIRs across small, medium and
large rooms.

## 2. MISSING, and it blocks everything downstream

**M-AILABS and MLAAD v5 are not on disk anywhere.** They are the bona fide and
fake sources we train on. Right now the project has held-out test sets, a
validation corpus and RIRs, and nothing to train on. See
`data/raw/train_sources/README.md`.

## 3. ReplayDF is a quarter downloaded

`aux/` is complete: all 110 UIDs with `RIR.wav`, sine sweeps, setup photo and an
`info.txt` naming the speaker and mic. So **gate G3 simulator validation can
proceed now** - it compares acoustic distributions and only needs the RIRs.

`wav/` has 28 of 110 UIDs: 12,967 files, 6,487 benign and 6,480 spoof, totalling
**32.9 h of the published 132.5 h**. Detection evaluation on ReplayDF should wait
for the rest, or be reported explicitly as a 28-UID subset.

Languages present: de, en, es, fr, it, pl. TTS systems: bark, vits, xtts_v1.1,
xtts_v2.

## 4. Redundant copies (kept, not deleted)

- `Audio (1).zip` was byte-identical to `Audio.zip`.
- `files.zip` held only `CLAUDE.md`, `PLAN.md` and `airgap-add.tar.gz`, all
  identical to the root copies.
- `sim_rir_16k.zip` was the same 60,611 simulated RIRs already inside
  `rirs_noises.zip`, entry for entry.

All three are in `data/archives/duplicates/` and can be removed to reclaim
about 190 MB.

## 5. Still downloading (in `C:\Users\admin\Downloads`, untouched)

- `A1.h5`, 1.07 GB, unidentified
- `BUT_ReverbDB_rel_19_06_RIR-Only.tgz.00`, 3.2 GB, part 00 of a multipart archive
- five `Unconfirmed *.crdownload` files, growing, ~25 GB and counting

These were still being written during the audit. Re-run the audit once they
finish and fold them into `data/raw/rir/`.

## 6. Contradictions with CLAUDE.md - reported, not resolved

Section 3 says to stop and report rather than pick a side.

1. **MIT IR Survey.** Section 3 records "davidscripka mirror: 16 kHz, 271 RIRs,
   ~15 MB". What is on disk is the original mcdermottlab `Audio.zip`: **270 RIRs
   at 32 kHz, 24-bit**, 17 MB. Either resample this copy to 16 kHz as a
   preprocessing step and correct section 3, or fetch the mirror section 3
   describes. Convolving 16 kHz speech with a 32 kHz RIR is a named failure mode
   in section 9.
2. **Project location.** Section 11 says root `D:\ag`, venv `D:\ag\venv`, data
   `D:\ag\data\raw`. Actual root is `C:\Users\admin\Documents\airgap-dd` with
   `.venv/` inside it, and D: is nearly empty. Section 11 needs updating, or the
   project needs moving.
3. **Python version.** Section 10 says 3.11. The venv is **3.12.10** and contains
   only `huggingface_hub` and its dependencies. Nothing in `requirements.txt` is
   installed yet, including torch, torchaudio, soundfile and pyroomacoustics.
