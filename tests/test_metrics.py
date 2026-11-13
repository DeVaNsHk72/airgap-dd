import numpy as np

from src.eval.metrics import compute_eer


def test_eer_perfect_separation() -> None:
    """Every bona fide score beats every spoof score: a threshold exists
    that gets everything right, so EER must be exactly 0."""
    target = np.array([5.0, 6.0, 7.0])
    nontarget = np.array([1.0, 2.0, 3.0])
    eer, _ = compute_eer(target, nontarget)
    assert eer == 0.0


def test_eer_identical_distributions() -> None:
    """Bona fide and spoof scores are drawn from the same values: no
    threshold does better than chance, so EER must be exactly 0.5."""
    target = np.array([1.0, 2.0, 3.0, 4.0])
    nontarget = np.array([1.0, 2.0, 3.0, 4.0])
    eer, _ = compute_eer(target, nontarget)
    assert eer == 0.5


if __name__ == "__main__":
    test_eer_perfect_separation()
    test_eer_identical_distributions()
    print("test_metrics: PASS")
