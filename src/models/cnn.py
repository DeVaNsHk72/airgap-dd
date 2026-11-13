"""Small raw-waveform CNN: the second model, after LightGBM's fast
feedback pass. First conv layer acts as a learnable filterbank directly
on the 64,600-sample window, the same convention RawNet2/AASIST use and
the same window length G1's evaluation and augment_gpu.py already share.

Aggressive early downsampling by design — the point is a small, fast
second model, not a reproduction of AASIST's full architecture (that's
what the vendored reference model in third_party/aasist/ is for).
"""

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=251, stride=4, padding=125),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),  # index 1 = bonafide, matching AASIST's convention
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (batch, 64600) -> (batch, 1, 64600)
        x = self.features(x)
        x = x.mean(dim=2)  # global average pool over time
        return self.classifier(x)
