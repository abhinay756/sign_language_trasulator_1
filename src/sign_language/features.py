from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np


LANDMARK_SIZE = 21 * 3
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def create_detector() -> mp_hands.Hands:
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )


def extract_landmarks(results) -> np.ndarray:
    """Return a translation/scale-normalized 63-value hand feature vector."""
    if not results.multi_hand_landmarks:
        return np.zeros(LANDMARK_SIZE, dtype=np.float32)

    points = np.array(
        [[landmark.x, landmark.y, landmark.z] for landmark in results.multi_hand_landmarks[0].landmark],
        dtype=np.float32,
    )
    points -= points[0:1]
    scale = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
    if scale > 1e-6:
        points /= scale
    return points.reshape(-1).astype(np.float32)


def detect(frame: np.ndarray, detector) -> tuple[np.ndarray, object]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = detector.process(rgb)
    rgb.flags.writeable = True
    return extract_landmarks(results), results


def draw_landmarks(frame: np.ndarray, results) -> None:
    if not results.multi_hand_landmarks:
        return
    for hand in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )
