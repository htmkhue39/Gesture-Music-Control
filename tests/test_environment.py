from importlib import metadata

import pytest

from src.landmarks import MODEL_PATH


def test_core_imports():
    import cv2, numpy, scipy, sklearn, matplotlib, pynput, mediapipe
    assert mediapipe.__version__ == "1.2.3"
    assert hasattr(cv2, "VideoCapture")

def test_model_file_present():
    assert MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 1_000_000


def test_plain_opencv_python_not_installed():
    # opencv-python next to opencv-contrib-python breaks cv2
    with pytest.raises(metadata.PackageNotFoundError):
        metadata.distribution("opencv-python")
