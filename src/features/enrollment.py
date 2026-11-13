"""Enrollment-conditioned features for the IoT / enrolled-speaker setting.

Deployment scenario: a voice device is enrolled to one person and holds a
few genuine clips of them. An attacker plays a synthetic clone of that
person through a speaker at the device. The question is not "is this
audio synthetic in general" (speaker-independent, and our experiments
stall near chance on it) but "is this my enrolled user, or a clone of
them?"

A device never has clones of its owner, so a per-speaker classifier is
unrealistic. Instead this builds features of the test utterance
*relative to* the enrolment centroid, so one global model serves any
enrolled speaker, including speakers never seen in training:

    [ test - enrol ,  cosine(test, enrol) ,  handcrafted(test) ]

The difference vector is the load-bearing part: a genuine utterance
should sit close to the enrolment centroid, while a clone should diverge
along synthesis-artifact directions even though it matches the voice.
"""

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def enrollment_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Mean of a speaker's enrolment embeddings. Robust enough at the
    handful-of-clips scale a real device would have."""
    return embeddings.mean(axis=0)


def build_conditioned_features(
    test_emb: np.ndarray,
    enrol_emb: np.ndarray,
    test_handcrafted: np.ndarray | None = None,
) -> np.ndarray:
    """One feature vector for a (test utterance, enrolled speaker) pair."""
    delta = test_emb - enrol_emb
    sim = np.array([cosine(test_emb, enrol_emb)], dtype=np.float32)
    parts = [delta.astype(np.float32), sim]
    if test_handcrafted is not None:
        parts.append(test_handcrafted.astype(np.float32))
    return np.concatenate(parts)


def build_matrix(
    test_embs: np.ndarray,
    enrol_embs_per_row: np.ndarray,
    test_handcrafted: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorised version for a whole set of trials.

    enrol_embs_per_row[i] is the enrolment centroid for the speaker that
    trial i is being scored against.
    """
    delta = (test_embs - enrol_embs_per_row).astype(np.float32)
    num = np.einsum("ij,ij->i", test_embs, enrol_embs_per_row)
    den = np.linalg.norm(test_embs, axis=1) * np.linalg.norm(enrol_embs_per_row, axis=1) + 1e-9
    sim = (num / den).astype(np.float32)[:, None]
    parts = [delta, sim]
    if test_handcrafted is not None:
        parts.append(test_handcrafted.astype(np.float32))
    return np.concatenate(parts, axis=1)
