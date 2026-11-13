import numpy as np
from numpy.typing import NDArray


def compute_det_curve(
    target_scores: NDArray[np.floating],
    nontarget_scores: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Detection error trade-off curve: FRR and FAR at every possible threshold.

    target_scores are bona fide, nontarget_scores are spoof. Convention:
    higher score = more bona fide-like. Get this backwards and EER silently
    comes out as (100 - true EER) instead of erroring.
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(
    target_scores: NDArray[np.floating],
    nontarget_scores: NDArray[np.floating],
) -> tuple[float, float]:
    """Equal error rate and the threshold it occurs at.

    Scores are discrete, so FRR and FAR almost never hit the same value
    exactly. We take the threshold where |FRR - FAR| is smallest and report
    the average of the two there.
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = float(np.mean((frr[min_index], far[min_index])))
    return eer, float(thresholds[min_index])
