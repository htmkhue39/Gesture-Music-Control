# Gesture-Music-Control

A student project that controls music with hand gestures seen by a webcam. This repo contains the shared setup: a pinned Python environment, a reusable hand-landmark detector (`src/landmarks.py`, built on the MediaPipe Tasks API), a webcam demo and tests.

## Requirements

- We use Python 3.12 (the environment is built and tested on it)
- A webcam
- git

## Setup on Windows (PowerShell)

1. Install Python 3.12 from https://www.python.org/downloads/ and tick **"Add python.exe to PATH"** in the installer.
2. Clone and create a virtual environment:

```powershell
git clone https://github.com/htmkhue39/Gesture-Music-Control.git
cd Gesture-Music-Control
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activate script, run this once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

3. Install the dependencies and run the tests:

```powershell
pip install -r requirements.txt
pytest
```

All tests should pass.

## Setup on macOS

```bash
git clone https://github.com/htmkhue39/Gesture-Music-Control.git
cd Gesture-Music-Control
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

All tests should pass.

## Run the demo

With the virtual environment activated:

```
python scripts/webcam_demo.py
python scripts/webcam_demo.py --camera 1 --hands 2
```

`--camera N` picks the camera (default 0), `--hands N` the maximum number of hands (default 1). Press `q` or `Esc` to quit. If the window's X button does not close it, press `q` or `Esc`.

## Check your setup (teammates)

Post in the group chat: a screenshot of the demo showing **your hand only (no face)**, and the output of `pytest`.

## Troubleshooting

- **Wrong or black camera:** try `python scripts/webcam_demo.py --camera 1`.
- **Camera busy:** close Teams, Zoom and anything else using the webcam.
- **macOS, "not authorized to capture video":** Terminal or VS Code has no camera permission. Open System Settings -> Privacy & Security -> Camera, enable the app, then restart it.
- **Windows, `py -3.12` not found:** install Python 3.12 (see Setup).
- **Never run `pip install opencv-python`.** It breaks `cv2`. If it was installed:
  ```
  pip uninstall opencv-python opencv-contrib-python
  pip install -r requirements.txt
  ```
- **`ModuleNotFoundError` / `AttributeError` about `mediapipe.solutions`:** that is old tutorial code. This mediapipe version has no `mp.solutions`; use `src/landmarks.py`.

## Using the landmarks in your own code

Run from the repo root (the image path is relative to it):

```python
import cv2
from src.landmarks import HandDetector

frame = cv2.imread("tests/data/hand.jpg")  # BGR image (sample photo in the repo)
with HandDetector(num_hands=1) as detector:
    for hand in detector.detect(frame):
        print(hand.handedness, hand.score, hand.landmarks.shape)  # (21, 3)
```

For video, use `HandDetector(video=True)` and pass `timestamp_ms` to `detect`. `hand.landmarks` are normalized image coordinates, `hand.world` are metres.

## Project layout

- `src/` - reusable code (`landmarks.py`: `HandDetector`)
- `scripts/` - runnable scripts (`webcam_demo.py`)
- `models/` - MediaPipe model file `hand_landmarker.task` (committed)
- `tests/` - pytest tests
- `docs/` - project documents
- `requirements.txt` - pinned dependencies
