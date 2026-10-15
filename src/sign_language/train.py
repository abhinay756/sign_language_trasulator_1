from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from .config import ARTIFACTS_DIR, DATA_DIR, load_config
from .dataset import GestureDataset
from .features import LANDMARK_SIZE
from .model import TemporalGestureClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the HandSign temporal classifier")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def discover_samples(classes: list[str]) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for label_index, label in enumerate(classes):
        files = sorted((DATA_DIR / label).glob("*.npy"))
        if len(files) < 2:
            raise SystemExit(f"Need at least 2 sequences for '{label}', found {len(files)}. Run collect-signs first.")
        samples.extend((path, label_index) for path in files)
    return samples


def accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for features, labels in loader:
            predictions = model(features.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
            total += len(labels)
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = load_config()
    classes = config["classes"]
    samples = discover_samples(classes)
    paths, labels = zip(*samples)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        list(paths), list(labels), test_size=0.2, random_state=args.seed, stratify=list(labels)
    )
    train_set = GestureDataset(list(zip(train_paths, train_labels)), augment=True)
    val_set = GestureDataset(list(zip(val_paths, val_labels)))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalGestureClassifier(LANDMARK_SIZE, len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ARTIFACTS_DIR / "gesture_model.pt"
    best_accuracy = -1.0

    print(f"Training {len(train_set)} samples, validating {len(val_set)} samples on {device}.")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for features, labels_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features.to(device)), labels_batch.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels_batch)
        val_accuracy = accuracy(model, val_loader, device)
        mean_loss = total_loss / len(train_set)
        print(f"epoch {epoch:03d}/{args.epochs}  loss={mean_loss:.4f}  val_accuracy={val_accuracy:.3f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "classes": classes,
                    "sequence_length": config["sequence_length"],
                    "input_size": LANDMARK_SIZE,
                    "state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
    print(f"Best validation accuracy: {best_accuracy:.3f}")
    print(f"Saved: {checkpoint_path}")
