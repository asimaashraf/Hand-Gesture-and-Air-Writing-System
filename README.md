<img width="1082" height="1015" alt="Screenshot 2026-06-06 151511" src="https://github.com/user-attachments/assets/34b11ed7-9c85-4828-8a6a-fab6f47a2b07" /># Hand Gesture and Air Writing System

A Python computer vision project for real-time hand gesture recognition and air writing using OpenCV, MediaPipe, TensorFlow Lite, and PyQt5.
![Uploading Screenshot 2026-06-06 151511.png…]()


## Features

- Real-time hand tracking with webcam
- Gesture recognition using trained keypoint classifier models
- Air writing mode with drawing output
- PyQt desktop writing interface
- Browser-based UI mockup served by the main app
- Model retraining script for keypoint classifier

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
└── tests/
```

## Requirements

- Python 3.10
- Webcam
- Windows, Linux, or macOS

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Setup

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

## Run

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

## Tests

Run unit tests:

```powershell
python -m unittest discover tests
```

## Model Files

The app needs these model files:

```text
model/keypoint_classifier/keypoint_classifier.tflite
model/keypoint_classifier/keypoint_classifier_label.csv
model/point_history_classifier/point_history_classifier.tflite
model/point_history_classifier/point_history_classifier_label.csv
```

Training data and retraining files are also included in the `model/` folder.

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

## License

Add your license here if you want to make the project open source.
