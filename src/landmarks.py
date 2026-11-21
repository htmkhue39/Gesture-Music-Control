"""Reusable hand landmark extraction on the MediaPipe Tasks API."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


@dataclass
class Hand:
    """One detected hand.

    Handedness assumes a mirrored (selfie) image, as MediaPipe does. For a
    non-mirrored image (e.g. a photo file), flip it with cv2.flip(img, 1)
    before detect() to get the same labels as the live demo.
    """

    landmarks: np.ndarray  # (21, 3) float32: normalized image x, y and relative z
    world: np.ndarray  # (21, 3) float32, metres
    handedness: str  # "Left" or "Right"
    score: float


class HandDetector:
    """Context manager wrapping HandLandmarker in image or video mode.

    A video-mode detector keeps tracking state between frames, so use one
    video-mode detector per clip (or camera stream).
    """

    def __init__(
        self,
        num_hands: int = 1,
        video: bool = False,
        min_detection_confidence: float = 0.5,
        model_path: Path = MODEL_PATH,
    ):
        self._video = video
        self._last_ts = -1
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mode,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._landmarker.close()

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> list[Hand]:
        """Return the hands found in a BGR image.

        Handedness assumes a mirrored (selfie) image; flip non-mirrored
        images with cv2.flip(img, 1) to match the live demo.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("empty image (None or size 0) - check the file path")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._video:
            if timestamp_ms is None:
                raise ValueError("timestamp_ms is required in video mode")
            ts = max(int(timestamp_ms), self._last_ts + 1)
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)

        hands = []
        for lm, world, handed in zip(
            result.hand_landmarks, result.hand_world_landmarks, result.handedness
        ):
            hands.append(
                Hand(
                    landmarks=np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32),
                    world=np.array([[p.x, p.y, p.z] for p in world], dtype=np.float32),
                    handedness=handed[0].category_name,
                    score=float(handed[0].score),
                )
            )
        return hands
