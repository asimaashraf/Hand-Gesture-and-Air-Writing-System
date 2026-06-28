# Hand Gesture and Air Writing System

A computer vision project for real-time hand gesture recognition and air writing.

This repository now contains two ways to use the project:

1. **Desktop Version (Python)** - the original OpenCV, MediaPipe, TensorFlow Lite, and PyQt application.
2. **Web Version (GitHub Pages)** - a browser-based version located in `Web_Version/`.

The original Python desktop application remains at the repository root and continues to use the existing `model/`, `utils/`, and training files.

<img width="1082" height="1015" alt="Screenshot 2026-06-06 151511" src="https://github.com/user-attachments/assets/d4ff0743-8e29-4b62-bbf7-481086a46a6d" />

## Desktop Version (Python)

### Features

- Real-time hand tracking with webcam
- Gesture recognition using trained keypoint classifier models
- Air writing mode with drawing output
- PyQt desktop writing interface
- Browser-based UI mockup served by the main app
- Model retraining script for keypoint classifier

### Requirements

- Python 3.10
- Webcam
- Windows, Linux, or macOS

### Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If you already have the `.venv` folder in this project, activate it:

```powershell
.\.venv\Scripts\activate
```

### Run

Run the main gesture recognition app:

```powershell
python main.py
```

Run the PyQt air writing UI:

```powershell
python pyqt_writing_ui.py
```

Use a different camera device:

```powershell
python main.py --device 1
```

Set camera size:

```powershell
python main.py --width 960 --height 540
```

### Tests

Run unit tests:

```powershell
python -m unittest discover tests
```

### Desktop Model Files

The desktop app needs these model files:

```text
model/keypoint_classifier/keypoint_classifier.tflite
model/keypoint_classifier/keypoint_classifier_label.csv
model/point_history_classifier/point_history_classifier.tflite
model/point_history_classifier/point_history_classifier_label.csv
```

Training data and retraining files are also included in the `model/` folder.

## Web Version (GitHub Pages)

The web version is fully contained in `Web_Version/`.

### Web Features

- Browser camera via `navigator.mediaDevices.getUserMedia()`
- Real-time hand detection with MediaPipe Hands
- Real-time gesture recognition using copied `.tflite` model assets when supported
- Heuristic fallback if browser TFLite loading is unavailable
- Real-time air writing
- Drawing canvas
- Color selection
- Eraser
- Clear canvas
- PNG download
- OCR with Tesseract.js
- Responsive desktop and mobile UI

### Web Files

```text
Web_Version/
├── index.html
├── style.css
├── web_app.js
├── camera.js
├── gesture.js
├── airwriting.js
├── ocr.js
├── models/
└── assets/
```

The web model files are copied into `Web_Version/models/`. The original desktop `model/` folder is not changed by the web version.

### Local Web Test

Camera APIs require a secure context. `localhost` is allowed for development:

```powershell
cd Web_Version
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

### GitHub Pages Deployment

This repository includes a GitHub Actions workflow at `.github/workflows/deploy-web-version.yml` that publishes only the `Web_Version/` folder to GitHub Pages.

After pushing to `main`, open the deployed web app at:

```text
https://asimaashraf.github.io/Hand-Gesture-and-Air-Writing-System/
```

In GitHub, make sure **Settings > Pages > Build and deployment > Source** is set to **GitHub Actions**. The workflow can also be run manually from the **Actions** tab.

No Flask, Python server, Docker, Render, Railway, paid hosting, or credit card is required for the deployed web version.

## Project Structure

```text
.
├── main.py
├── pyqt_writing_ui.py
├── retrain_keypoint_classifier.py
├── requirements.txt
├── ui_mockup.html
├── model/
├── utils/
├── tests/
└── Web_Version/
```

## GitHub Notes

Do not upload virtual environments, cache files, generated output images, or backup files. These are ignored by `.gitignore`:

```text
.venv/
myenv/
.idea/
__pycache__/
output/
*.pyc
*.bak_*
```
