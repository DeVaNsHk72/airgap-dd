# G1: EER implementation validation

**Gate:** our EER implementation must reproduce AASIST's published
ASVspoof19-LA eval EER within 0.5 points.

## Method

Ran the authors' own pretrained checkpoint (`third_party/aasist/weights/AASIST.pth`,
from [clovaai/aasist](https://github.com/clovaai/aasist), MIT licensed) over
the full ASVspoof2019 LA eval set, scored with `src/eval/metrics.py`.

- Score convention: `output[:, 1]` (bona fide logit), matching the official
  repo's `produce_evaluation_file` exactly.
- Preprocessing: truncate to first 64,600 samples (~4.04s @ 16kHz), or
  tile-repeat if shorter. Matches the official `data_utils.pad`.
- 7,355 bona fide trials, 63,882 spoof trials (matches the known
  ASVspoof2019-LA eval split composition).

## Result

| | EER |
|---|---|
| Published (AASIST paper, Table 2) | 0.83% |
| Ours | 0.83% |
| Difference | 0.00 points |

**G1 PASS.** `compute_eer` / `compute_det_curve` are validated against a
real published result, not just synthetic sanity cases (see
`tests/test_metrics.py`).

Date run: 2026-09-07
