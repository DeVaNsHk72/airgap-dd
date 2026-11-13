"""Full replay channel: chains the four stages in the order sound
actually travels — speaker coloration, then distortion from being driven
through the speaker, then the room, then whatever codec the recording
passes through afterward.

This is the randomized simulated channel only. It does NOT include
EchoFake's fixed final postprocessing (-23 LUFS normalize, then MP3
64kbps/16kHz/mono, applied by EchoFake to every one of its own subsets —
CLAUDE.md section 3). That's a separate, non-randomized step that belongs
at the training-data-generation/config level, only when targeting an
EchoFake-matched run — mixing it into this general-purpose simulator
would make the same code silently wrong for a ReplayDF-targeted run.
"""

from pathlib import Path

import numpy as np

from src.sim.codec import random_codec_roundtrip
from src.sim.nonlinear import random_nonlinear_distortion
from src.sim.rir import random_room_response
from src.sim.speaker_response import random_speaker_response


def simulate_replay_channel(
    x: np.ndarray, sr: int, rng: np.random.Generator, rir_paths: list[Path]
) -> np.ndarray:
    y = random_speaker_response(x, sr, rng)
    y = random_nonlinear_distortion(y, rng)
    y = random_room_response(y, sr, rng, rir_paths)
    y = random_codec_roundtrip(y, sr, rng)
    return y
