from __future__ import annotations

import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .features import LANDMARK_SIZE
from .dataset import GestureDataset
from .model import TemporalGestureClassifier
from .train import discover_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained HandSign checkpoint")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    classes = checkpoint["classes"]
    samples = discover_samples(classes)
    loader = DataLoader(GestureDataset(samples), batch_size=32)
    model = TemporalGestureClassifier(LANDMARK_SIZE, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for features, labels in loader:
            actual.extend(labels.tolist())
            predicted.extend(model(features).argmax(dim=1).tolist())
    print(classification_report(actual, predicted, target_names=classes, zero_division=0))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(actual, predicted, labels=list(range(len(classes)))))
