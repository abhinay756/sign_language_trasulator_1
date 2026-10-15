from __future__ import annotations

import torch
from torch import nn


class TemporalGestureClassifier(nn.Module):
    """Small sequence model suitable for CPU webcam inference."""

    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 128):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(inputs)
        encoded, _ = self.encoder(projected)
        summary = torch.cat([encoded[:, -1], encoded.mean(dim=1)], dim=1)
        return self.head(summary)
