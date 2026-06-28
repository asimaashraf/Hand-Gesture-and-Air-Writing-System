# Web Version

This folder is a standalone browser implementation of the hand gesture and air writing project. It does not use Flask, Python, Docker, Render, Railway, or any local server after deployment.

## Files

- `index.html` - GitHub Pages entry point.
- `style.css` - responsive desktop/mobile UI.
- `camera.js` - browser camera via `navigator.mediaDevices.getUserMedia()`.
- `gesture.js` - MediaPipe landmark preprocessing and browser gesture classification.
- `airwriting.js` - drawing canvas, colors, eraser, clear, and PNG export.
- `ocr.js` - browser OCR through Tesseract.js.
- `models/` - copied model and label files from the desktop project.

## Gesture Controls

- Pointer/index finger: draw in Writing mode.
- Open palm: clear canvas.
- Closed fist: stop drawing.
- Two fingers/Victory: selection posture; use the UI controls for precise tool changes.
- Toolbar buttons: color, eraser, clear, PNG download, OCR.

## Local Test

Camera APIs require a secure context. `localhost` is allowed for testing:

```powershell
cd Web_Version
python -m http.server 8000
```

Open `http://localhost:8000`.

## GitHub Pages Deployment

1. Commit the `Web_Version` folder to your GitHub repository.
2. In GitHub, open the repository settings.
3. Go to **Pages**.
4. Set **Source** to **Deploy from a branch**.
5. Select your branch, usually `main`.
6. Select `/root` as the folder if you want the whole repository published.
7. Open the published URL with `/Web_Version/` at the end:

```text
https://YOUR_USERNAME.github.io/YOUR_REPOSITORY/Web_Version/
```

For a cleaner URL, you can publish only the `Web_Version` contents from a separate branch or repository.

## Notes

The original desktop files and original `model/` folder are not modified. The `.tflite` and label files inside `Web_Version/models/` are copied assets kept separate from the desktop model files.
