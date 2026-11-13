"""G1 gate: does our EER implementation reproduce AASIST's published
ASVspoof19-LA number? Runs the authors' own pretrained checkpoint
(third_party/aasist/) over the eval set and scores it with our metrics.py.

Published target: 0.83% EER (AASIST paper, Table 2).
Pass condition: within 0.5 points of that.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "aasist"))
from AASIST import Model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import compute_eer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_FLAC_DIR = REPO_ROOT / "data/raw/validation_only/asvspoof19_la/LA/ASVspoof2019_LA_eval/flac"
EVAL_PROTOCOL = (
    REPO_ROOT
    / "data/raw/validation_only/asvspoof19_la/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
)
WEIGHTS_PATH = REPO_ROOT / "third_party/aasist/weights/AASIST.pth"

# Copied verbatim from the authors' config/AASIST.conf, not something we tune.
MODEL_CONFIG = {
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}
PUBLISHED_EER = 0.83


def pad_to_fixed_length(x: np.ndarray, max_len: int = 64600) -> np.ndarray:
    """Truncate to the first max_len samples, or tile-repeat if shorter.
    Matches the official AASIST repo's data_utils.pad exactly."""
    if x.shape[0] >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x.shape[0]) + 1
    return np.tile(x, num_repeats)[:max_len]


class AsvspoofEvalSet(Dataset):
    def __init__(self, protocol_path: Path, flac_dir: Path):
        self.utt_ids: list[str] = []
        self.labels: list[str] = []
        with open(protocol_path) as f:
            for line in f:
                _, utt_id, _, _, label = line.strip().split(" ")
                self.utt_ids.append(utt_id)
                self.labels.append(label)
        self.flac_dir = flac_dir

    def __len__(self) -> int:
        return len(self.utt_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        utt_id = self.utt_ids[idx]
        x, _ = sf.read(self.flac_dir / f"{utt_id}.flac")
        x = pad_to_fixed_length(x.astype(np.float32))
        return torch.from_numpy(x), utt_id


def load_model(device: torch.device) -> torch.nn.Module:
    model = Model(MODEL_CONFIG).to(device)
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset = AsvspoofEvalSet(EVAL_PROTOCOL, EVAL_FLAC_DIR)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, persistent_workers=True)
    model = load_model(device)

    scores: dict[str, float] = {}
    with torch.no_grad():
        for batch_x, utt_ids in loader:
            batch_x = batch_x.to(device)
            _, batch_out = model(batch_x)
            batch_scores = batch_out[:, 1].cpu().numpy()
            for utt_id, score in zip(utt_ids, batch_scores):
                scores[utt_id] = float(score)

    label_by_id = dict(zip(dataset.utt_ids, dataset.labels))
    target = np.array([scores[u] for u in dataset.utt_ids if label_by_id[u] == "bonafide"])
    nontarget = np.array([scores[u] for u in dataset.utt_ids if label_by_id[u] == "spoof"])
    print(f"bonafide trials: {target.size}  spoof trials: {nontarget.size}")

    eer, threshold = compute_eer(target, nontarget)
    eer_pct = eer * 100
    print(f"EER: {eer_pct:.2f}%  (threshold {threshold:.4f})")
    print(f"published: {PUBLISHED_EER:.2f}%  diff: {abs(eer_pct - PUBLISHED_EER):.2f} points")
    if abs(eer_pct - PUBLISHED_EER) <= 0.5:
        print("G1 PASS")
    else:
        print("G1 FAIL — do not trust downstream numbers until this is fixed")


if __name__ == "__main__":
    main()
