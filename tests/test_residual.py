"""Regression test: Burg's method LPC can produce exploding coefficients
on narrow-band/resonant content (a documented "pole splitting" failure
mode), and the simulator's own resonances make this more likely on
augmented audio than clean audio. _fit_lpc_robust must catch this via
order fallback, not let it propagate as NaN/Inf into a training feature."""

import numpy as np

from src.features.residual import compute_lp_residual, extract_residual_features


def test_residual_finite_on_narrowband_signal() -> None:
    """A near-pure sine tone is exactly the kind of narrow-band content
    that ill-conditions a high-order LPC fit."""
    sr = 16000
    t = np.arange(sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    residual = compute_lp_residual(tone)
    assert np.isfinite(residual).all()
    assert np.abs(residual).max() < 10.0  # sane bound, not 1e16

    feats = extract_residual_features(tone)
    assert all(np.isfinite(v) for v in feats.values())


def test_residual_finite_on_real_speech() -> None:
    from pathlib import Path

    from src.data.loaders import load_audio

    wav = next(Path("data/raw/train_sources/mailabs/en_US").rglob("*.wav"))
    x = load_audio(wav)
    feats = extract_residual_features(x)
    assert all(np.isfinite(v) for v in feats.values())


if __name__ == "__main__":
    test_residual_finite_on_narrowband_signal()
    test_residual_finite_on_real_speech()
    print("test_residual: PASS")
