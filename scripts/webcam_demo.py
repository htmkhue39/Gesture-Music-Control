"""Webcam demo: mirrored live view with hand landmarks drawn on top."""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections

from src.landmarks import Hand, HandDetector

WINDOW = "Gesture Music Control"
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def open_camera(index: int = 0) -> cv2.VideoCapture:
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {index}. Try another index with --camera N, "
            "and check that camera permission is granted to your terminal/IDE "
            "and that no other app is using the camera."
        )
    return cap


def draw_hands(frame_bgr: np.ndarray, hands: list[Hand]) -> None:
    h, w = frame_bgr.shape[:2]
    for hand in hands:
        pts = [(int(x * w), int(y * h)) for x, y, _ in hand.landmarks]
        for c in HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(frame_bgr, pts[c.start], pts[c.end], GREEN, 2)
        for p in pts:
            cv2.circle(frame_bgr, p, 4, RED, -1)
        wx, wy = pts[0]
        cv2.putText(frame_bgr, f"{hand.handedness} {hand.score:.2f}", (wx, max(wy - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--hands", type=int, default=1, help="max hands to detect (default 1)")
    args = parser.parse_args(argv)

    try:
        cap = open_camera(args.camera)
    except RuntimeError as e:
        print(e)
        return 1

    times: deque[float] = deque(maxlen=30)
    try:
        with HandDetector(video=True, num_hands=args.hands) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print(
                        "Camera stopped sending frames. Close other apps using the camera "
                        "(Teams/Zoom), try --camera N, and on Windows enable Settings > "
                        "Privacy & security > Camera > Let desktop apps access your camera."
                    )
                    break
                frame = cv2.flip(frame, 1)
                hands = detector.detect(frame, timestamp_ms=int(time.monotonic() * 1000))
                draw_hands(frame, hands)

                times.append(time.monotonic())
                if len(times) > 1 and times[-1] > times[0]:
                    fps = (len(times) - 1) / (times[-1] - times[0])
                    cv2.putText(frame, f"FPS {fps:.1f}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
                cv2.putText(frame, "q / Esc to quit", (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
