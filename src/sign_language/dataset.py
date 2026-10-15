from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GestureDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        sequence = np.load(path).astype(np.float32)
        if self.augment:
            sequence = sequence + np.random.normal(0, 0.015, sequence.shape).astype(np.float32)
            if np.random.random() < 0.15:
                sequence[np.random.randint(0, len(sequence))] = 0
        return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.long)
