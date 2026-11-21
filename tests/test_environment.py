from pathlib import Path

def test_core_imports():
    import cv2, numpy, scipy, sklearn, matplotlib, pynput, mediapipe
    assert mediapipe.__version__ == "1.2.3"
    assert hasattr(cv2, "VideoCapture")

def test_model_file_present():
    p = Path("models/hand_landmarker.task")
    assert p.is_file() and p.stat().st_size > 1_000_000
