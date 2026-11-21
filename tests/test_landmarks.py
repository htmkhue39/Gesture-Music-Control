from pathlib import Path

import cv2
import numpy as np

from src.landmarks import HandDetector, Hand

FIXTURE = Path(__file__).parent / "data" / "hand.jpg"


def test_no_hand_in_blank_image():
    with HandDetector() as d:
        assert d.detect(np.zeros((480, 640, 3), np.uint8)) == []


def test_detects_hand_in_fixture():
    img = cv2.imread(str(FIXTURE))
    with HandDetector() as d:
        hands = d.detect(img)
    assert len(hands) == 1
    h = hands[0]
    assert isinstance(h, Hand)
    assert h.landmarks.shape == (21, 3) and h.world.shape == (21, 3)
    assert ((h.landmarks[:, :2] >= -0.1) & (h.landmarks[:, :2] <= 1.1)).all()
    assert h.handedness in ("Left", "Right") and 0 < h.score <= 1


def test_video_mode_tolerates_repeated_timestamp():
    img = cv2.imread(str(FIXTURE))
    with HandDetector(video=True) as d:
        d.detect(img, timestamp_ms=100)
        d.detect(img, timestamp_ms=100)  # must not raise
        assert len(d.detect(img, timestamp_ms=101)) == 1
