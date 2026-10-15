from __future__ import annotations

import argparse
import time
from collections import Counter, deque

import cv2
import numpy as np
import torch

from .features import LANDMARK_SIZE, create_detector, detect, draw_landmarks
from .model import TemporalGestureClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time HandSign recognition")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    classes = checkpoint["classes"]
    sequence_length = int(checkpoint["sequence_length"])
    model = TemporalGestureClassifier(LANDMARK_SIZE, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Try --camera 1 or another device index.")

    sequence = deque(maxlen=sequence_length)
    recent_predictions = deque(maxlen=7)
    last_label = "Waiting for a hand"
    last_confidence = 0.0
    previous_time = time.perf_counter()

    with create_detector() as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            features, results = detect(frame, detector)
            draw_landmarks(frame, results)
            sequence.append(features)
            if len(sequence) == sequence_length:
                batch = torch.from_numpy(np.asarray(sequence, dtype=np.float32)).unsqueeze(0)
                with torch.no_grad():
                    probabilities = torch.softmax(model(batch), dim=1)[0]
                index = int(probabilities.argmax())
                confidence = float(probabilities[index])
                recent_predictions.append(index)
                stable_index, stable_count = Counter(recent_predictions).most_common(1)[0]
                if stable_count >= 4 and confidence >= args.threshold:
                    last_label = classes[stable_index]
                    last_confidence = confidence

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            cv2.rectangle(frame, (0, 0), (420, 100), (20, 20, 20), -1)
            cv2.putText(frame, f"Sign: {last_label}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 230, 120), 2)
            cv2.putText(frame, f"Confidence: {last_confidence:.0%}   FPS: {fps:.1f}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, "Q: quit", (15, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("HandSign real-time recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
