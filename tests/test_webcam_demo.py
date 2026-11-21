import numpy as np, pytest, cv2
from scripts import webcam_demo as wd
from src.landmarks import Hand

def _hand():
    pts = np.column_stack([np.linspace(.2, .8, 21), np.linspace(.2, .8, 21), np.zeros(21)]).astype(np.float32)
    return Hand(landmarks=pts, world=pts.copy(), handedness="Right", score=0.9)

def test_draw_hands_changes_pixels():
    f = np.zeros((480, 640, 3), np.uint8)
    wd.draw_hands(f, [_hand()])
    assert f.any()

def test_draw_hands_no_hands_is_noop():
    f = np.zeros((480, 640, 3), np.uint8)
    wd.draw_hands(f, [])
    assert not f.any()

class _FakeCap:
    def __init__(self, *args): self.args = args
    def isOpened(self): return False

def test_open_camera_uses_dshow_on_windows(monkeypatch):
    monkeypatch.setattr(wd.sys, "platform", "win32")
    monkeypatch.setattr(wd.cv2, "VideoCapture", _FakeCap)
    with pytest.raises(RuntimeError) as e:
        wd.open_camera(1)
    assert "--camera" in str(e.value)

def test_open_camera_backend_arg(monkeypatch):
    seen = {}
    class Cap(_FakeCap):
        def __init__(self, *a): seen["a"] = a
        def isOpened(self): return True
    monkeypatch.setattr(wd.cv2, "VideoCapture", Cap)
    monkeypatch.setattr(wd.sys, "platform", "win32")
    wd.open_camera(1); assert seen["a"] == (1, cv2.CAP_DSHOW)
    monkeypatch.setattr(wd.sys, "platform", "darwin")
    wd.open_camera(0); assert seen["a"] == (0,)
