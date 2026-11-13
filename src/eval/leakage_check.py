"""G4 gate: no ReplayDF or EchoFake path in any training manifest.

Rule 1 (CLAUDE.md): never train on ReplayDF or EchoFake, not even for a
sanity check. Re-run this before every reported result, not just once —
a future change to the manifest-building code could reintroduce a leak
silently.
"""

from pathlib import Path

from src.data.protocols import Utterance

HELDOUT_MARKERS = ("heldout", "replaydf", "echofake")


def assert_no_heldout_leakage(utterances: list[Utterance]) -> None:
    for u in utterances:
        path_lower = str(u.path).lower()
        if any(marker in path_lower for marker in HELDOUT_MARKERS):
            raise AssertionError(f"G4 FAILED: held-out path in training manifest: {u.path}")
