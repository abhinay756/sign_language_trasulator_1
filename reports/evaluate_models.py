from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sign_language.features import LANDMARK_SIZE  # noqa: E402
from sign_language.model import TemporalGestureClassifier  # noqa: E402

DATA_DIR = ROOT / "data" / "landmarks"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
ASSET_DIR = REPORT_DIR / "assets"
CLASSES = ["A", "B", "C", "HI", "BYE", "THANK_YOU", "YES", "NO"]
SEED = 7


def load_data() -> tuple[np.ndarray, np.ndarray]:
    sequences, labels = [], []
    for label_index, label in enumerate(CLASSES):
        for path in sorted((DATA_DIR / label).glob("*.npy")):
            sequence = np.load(path).astype(np.float32)
            if sequence.shape != (30, LANDMARK_SIZE):
                raise ValueError(f"Unexpected shape in {path}: {sequence.shape}")
            sequences.append(sequence)
            labels.append(label_index)
    return np.asarray(sequences), np.asarray(labels)


class LandmarkCNN(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(48, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.network(x).flatten(1))


def apply_ry(state: torch.Tensor, angle: torch.Tensor, qubit: int) -> torch.Tensor:
    zero = [i for i in range(16) if ((i >> qubit) & 1) == 0]
    one = [i | (1 << qubit) for i in zero]
    a, b = state[:, zero], state[:, one]
    c, s = torch.cos(angle / 2).unsqueeze(1), torch.sin(angle / 2).unsqueeze(1)
    result = state.clone()
    result[:, zero] = c * a - s * b
    result[:, one] = s * a + c * b
    return result


def apply_cnot(state: torch.Tensor, control: int, target: int) -> torch.Tensor:
    result = state.clone()
    for i in range(16):
        if ((i >> control) & 1) == 1:
            result[:, i ^ (1 << target)] = state[:, i]
    return result


class StateVectorQNN(nn.Module):
    """Four-qubit variational circuit simulated with a differentiable state vector."""

    def __init__(self, classes: int):
        super().__init__()
        self.rotation_angles = nn.Parameter(torch.zeros(2, 4))
        self.classifier = nn.Linear(4, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = torch.zeros((x.shape[0], 16), dtype=torch.complex64, device=x.device)
        state[:, 0] = 1.0 + 0.0j
        for layer in range(2):
            for qubit in range(4):
                state = apply_ry(state, x[:, qubit] + self.rotation_angles[layer, qubit], qubit)
            for qubit in range(4):
                state = apply_cnot(state, qubit, (qubit + 1) % 4)
        probabilities = state.abs().square()
        measurements = []
        for qubit in range(4):
            signs = torch.tensor([1.0 if ((i >> qubit) & 1) == 0 else -1.0 for i in range(16)], device=x.device)
            measurements.append((probabilities * signs).sum(dim=1))
        return self.classifier(torch.stack(measurements, dim=1))


def train_torch(model, train_x, train_y, val_x, val_y, input_kind, epochs=30):
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()
    train_features = torch.from_numpy(train_x).float()
    train_labels = torch.from_numpy(train_y).long()
    val_features = torch.from_numpy(val_x).float()
    val_labels = torch.from_numpy(val_y).long()
    history = {"train_loss": [], "val_accuracy": []}
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_features)
        loss = loss_fn(logits, train_labels)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = model(val_features).argmax(dim=1).numpy()
        history["train_loss"].append(float(loss.item()))
        history["val_accuracy"].append(float(accuracy_score(val_y, predictions)))
    model.eval()
    with torch.no_grad():
        final_predictions = model(val_features).argmax(dim=1).numpy()
    return final_predictions, history


def save_plots(results: dict, histories: dict, matrices: dict) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 12, "axes.labelsize": 10})

    names = list(results)
    values = [results[name] * 100 for name in names]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.bar(names, values, color=["#1d4ed8", "#0f766e", "#7c3aed", "#d97706"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Model validation accuracy")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "accuracy_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.4))
    for ax, name in zip(axes.flat, names):
        matrix = matrices[name]
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=7)
        for row in range(len(CLASSES)):
            for col in range(len(CLASSES)):
                if matrix[row, col]:
                    ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=7)
    fig.suptitle("Validation confusion matrices")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "confusion_matrices.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    for ax, name in zip(axes, ["ANN", "CNN", "QNN simulation"]):
        history = histories[name]
        ax.plot(history["train_loss"], label="train loss", color="#1d4ed8")
        ax2 = ax.twinx()
        ax2.plot(history["val_accuracy"], label="val accuracy", color="#d97706")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)
    fig.suptitle("Training loss and validation accuracy")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "training_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.0, 3.6))
    stages = ["Webcam", "Landmarks", "Sequence", "Model", "Prediction"]
    x = np.arange(len(stages))
    ax.scatter(x, np.zeros(len(x)), s=1500, color=["#0f766e", "#1d4ed8", "#7c3aed", "#d97706", "#b91c1c"])
    for i, stage in enumerate(stages):
        ax.text(i, 0, str(i + 1), ha="center", va="center", color="white", fontsize=13, weight="bold")
        ax.text(i, -0.30, stage, ha="center", va="center", color="#1f2937", fontsize=9, weight="bold")
        if i < len(stages) - 1:
            ax.annotate("", xy=(i + 0.78, 0), xytext=(i + 0.22, 0), arrowprops={"arrowstyle": "->", "lw": 2, "color": "#334155"})
    ax.set_xlim(-0.6, len(stages) - 0.4)
    ax.set_ylim(-0.55, 0.55)
    ax.axis("off")
    ax.set_title("HandSign system pipeline")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "pipeline.png", dpi=180, transparent=False)
    plt.close(fig)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    sequences, labels = load_data()
    train_seq, val_seq, train_y, val_y = train_test_split(
        sequences, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    # Current project model: temporal GRU checkpoint.
    checkpoint = torch.load(ARTIFACT_DIR / "gesture_model.pt", map_location="cpu", weights_only=False)
    gru = TemporalGestureClassifier(LANDMARK_SIZE, len(CLASSES))
    gru.load_state_dict(checkpoint["state_dict"])
    gru.eval()
    with torch.no_grad():
        gru_pred = gru(torch.from_numpy(val_seq)).argmax(dim=1).numpy()

    # ANN baseline: flattened sequence.
    ann = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=180, early_stopping=True, random_state=SEED, validation_fraction=0.15)
    ann.fit(train_seq.reshape(len(train_seq), -1), train_y)
    ann_pred = ann.predict(val_seq.reshape(len(val_seq), -1))
    ann_history = {"train_loss": [float(x) for x in ann.loss_curve_], "val_accuracy": [float(x) for x in ann.validation_scores_]}

    cnn_pred, cnn_history = train_torch(
        LandmarkCNN(len(CLASSES)),
        train_seq.transpose(0, 2, 1).reshape(len(train_seq), 3, 30, 21),
        train_y,
        val_seq.transpose(0, 2, 1).reshape(len(val_seq), 3, 30, 21),
        val_y,
        "cnn",
        epochs=30,
    )

    pca = PCA(n_components=4, random_state=SEED)
    train_flat = train_seq.reshape(len(train_seq), -1)
    val_flat = val_seq.reshape(len(val_seq), -1)
    train_q = pca.fit_transform(train_flat).astype(np.float32)
    val_q = pca.transform(val_flat).astype(np.float32)
    scale = np.maximum(np.abs(train_q).max(axis=0), 1e-6)
    train_q = np.clip(train_q / scale, -1, 1) * np.pi
    val_q = np.clip(val_q / scale, -1, 1) * np.pi
    qnn_pred, qnn_history = train_torch(StateVectorQNN(len(CLASSES)), train_q, train_y, val_q, val_y, "qnn", epochs=45)

    predictions = {"GRU project model": gru_pred, "ANN": ann_pred, "CNN": cnn_pred, "QNN simulation": qnn_pred}
    results = {name: float(accuracy_score(val_y, pred)) for name, pred in predictions.items()}
    matrices = {name: confusion_matrix(val_y, pred, labels=list(range(len(CLASSES)))) for name, pred in predictions.items()}
    histories = {"ANN": ann_history, "CNN": cnn_history, "QNN simulation": qnn_history}
    save_plots({"GRU project model": results["GRU project model"], "ANN": results["ANN"], "CNN": results["CNN"], "QNN simulation": results["QNN simulation"]}, histories, matrices)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "metrics.json").write_text(json.dumps({"classes": CLASSES, "sample_counts": {label: int((labels == i).sum()) for i, label in enumerate(CLASSES)}, "results": results}, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(sequences), "results": results}, indent=2))


if __name__ == "__main__":
    main()
