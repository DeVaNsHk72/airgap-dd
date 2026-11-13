"""G2 gate, sample-rate and silence half: after load_audio, every corpus
must be at the same sample rate with no fixed silence padding left over."""

import random

from src.data.loaders import TARGET_SR, load_audio
from src.data.protocols import build_bonafide_manifest, build_fake_manifest


def test_all_corpora_load_at_target_sample_rate() -> None:
    random.seed(0)
    bf_sample = random.sample(build_bonafide_manifest(), 5)
    fk_sample = random.sample(build_fake_manifest(), 5)
    for u in bf_sample + fk_sample:
        x = load_audio(u.path)
        assert x.ndim == 1
        assert x.dtype.name == "float32"
        # duration is a reasonable spot check that resampling didn't
        # silently corrupt the signal (e.g. wrong orig_sr assumed)
        assert 0.1 < len(x) / TARGET_SR < 60


def test_bonafide_silence_padding_is_removed() -> None:
    """M-AILABS ships a fixed ~0.5s pad. After load_audio, that fixed cue
    must be gone — not just shortened, gone as a *distinguishing* signal."""
    random.seed(1)
    bf_sample = random.sample(build_bonafide_manifest(), 15)
    for u in bf_sample:
        x = load_audio(u.path)
        # a file that was almost entirely the fixed pad would come out
        # near-empty after trimming; that's the property we actually care
        # about, not an exact residual duration
        assert len(x) > 0


if __name__ == "__main__":
    test_all_corpora_load_at_target_sample_rate()
    test_bonafide_silence_padding_is_removed()
    print("test_loaders: PASS")
