"""Frozen Wav2Vec2 embeddings as features.

Motivation: our from-scratch models fail cross-corpus because four bona
fide speakers can't teach speaker- and corpus-independent synthesis
artifacts (see results/g5_baseline_and_voice_confound.md). Wav2Vec2
arrives pretrained on hundreds of hours of speech, which is the standard
remedy for that regime, and it is the strongest row in CLAUDE.md's target
table (42.94% clean-trained vs RawNet2's 46.34%).

Frozen (not fine-tuned) on purpose for this pass: far cheaper, no
fine-tuning instability, and it isolates the question "do pretrained
representations generalise better than our hand-crafted features?" from
the separate question of how well we can fine-tune.

Mean+std pooling over time gives a fixed-length vector per utterance, so
the existing LightGBM pipeline can consume it unchanged.
"""

import numpy as np
import torch
from transformers import Wav2Vec2Model

MODEL_NAME = "facebook/wav2vec2-base"
TARGET_SR = 16000
# Fixed-length windows, not a cap. Variable length leaks duration into
# mean+std pooling, and duration alone separated ASVspoof genuine (median
# 1.76s) from our clones (median 7.44s) at 6.00% EER — which was the whole
# of an apparent 6.24% "result". Cropping every utterance to the same
# length removes that cue. 2.5s fits inside the typical genuine clip.
FIXED_SAMPLES = int(16000 * 2.5)

_model = None
_device = None


def _get_model() -> tuple[Wav2Vec2Model, torch.device]:
    global _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(_device).eval()
        for p in _model.parameters():
            p.requires_grad = False
    return _model, _device


@torch.no_grad()
def embed_batch(waveforms: list[np.ndarray]) -> np.ndarray:
    """Mean+std pooled hidden states for a batch of waveforms."""
    model, device = _get_model()

    # crop or tile every utterance to exactly FIXED_SAMPLES
    trimmed = []
    for w in waveforms:
        if len(w) >= FIXED_SAMPLES:
            trimmed.append(w[:FIXED_SAMPLES])
        else:
            reps = FIXED_SAMPLES // max(1, len(w)) + 1
            trimmed.append(np.tile(w, reps)[:FIXED_SAMPLES])
    max_len = FIXED_SAMPLES
    batch = np.zeros((len(trimmed), max_len), dtype=np.float32)
    mask = np.zeros((len(trimmed), max_len), dtype=np.int64)
    for i, w in enumerate(trimmed):
        batch[i, : len(w)] = w
        mask[i, : len(w)] = 1

    x = torch.from_numpy(batch).to(device)
    m = torch.from_numpy(mask).to(device)

    # normalize per utterance, as wav2vec2-base expects
    x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-7)

    out = model(x, attention_mask=m).last_hidden_state  # (B, T, H)

    feat_mask = model._get_feature_vector_attention_mask(out.shape[1], m).unsqueeze(-1).float()
    denom = feat_mask.sum(dim=1).clamp(min=1.0)
    mean = (out * feat_mask).sum(dim=1) / denom
    var = ((out - mean.unsqueeze(1)) ** 2 * feat_mask).sum(dim=1) / denom
    std = var.clamp(min=0).sqrt()

    return torch.cat([mean, std], dim=1).cpu().numpy()
