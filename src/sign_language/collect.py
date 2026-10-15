from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .config import DATA_DIR, load_config
from .features import create_detector, detect, draw_landmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect webcam landmark sequences for every configured sign")
    parser.add_argument("--samples", type=int, default=40, help="sequences to collect per label")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--labels", nargs="*", help="optional subset such as A B HI")
    return parser.parse_args()


def next_index(folder: Path) -> int:
    existing = [int(path.stem) for path in folder.glob("*.npy") if path.stem.isdigit()]
    return max(existing, default=-1) + 1


def main() -> None:
    args = parse_args()
    config = load_config()
    labels = args.labels or config["classes"]
    unknown = sorted(set(labels) - set(config["classes"]))
    if unknown:
        raise SystemExit(f"Unknown labels: {', '.join(unknown)}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get("camera_width", 960))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get("camera_height", 540))
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Try --camera 1 or another device index.")

    print("Collection started. SPACE records a sequence; Q quits.")
    with create_detector() as detector:
        stop = False
        for label in labels:
            folder = DATA_DIR / label
            folder.mkdir(parents=True, exist_ok=True)
            start_index = next_index(folder)
            for offset in range(args.samples):
                index = start_index + offset
                frames: list[np.ndarray] = []
                armed = False
                while not armed and not stop:
                    ok, frame = cap.read()
                    if not ok:
                        stop = True
                        break
                    features, results = detect(frame, detector)
                    draw_landmarks(frame, results)
                    cv2.putText(frame, f"Label: {label} | sample {offset + 1}/{args.samples}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 40), 2)
                    cv2.putText(frame, "Hold SPACE to record | Q quits", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    cv2.imshow("HandSign collection", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        stop = True
                    elif key == 32:
                        armed = True

                if stop:
                    break

                for frame_number in range(config["sequence_length"]):
                    ok, frame = cap.read()
                    if not ok:
                        stop = True
                        break
                    features, results = detect(frame, detector)
                    draw_landmarks(frame, results)
                    frames.append(features)
                    progress = frame_number + 1
                    cv2.putText(frame, f"Recording {label}: {progress}/{config['sequence_length']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2)
                    cv2.imshow("HandSign collection", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        stop = True
                        break

                if stop:
                    break
                np.save(folder / f"{index:04d}.npy", np.asarray(frames, dtype=np.float32))
                time.sleep(0.15)

            if stop:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved landmark sequences under {DATA_DIR}")
