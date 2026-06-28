class WebHandGestureApp {
  constructor() {
    this.video = document.getElementById("cameraVideo");
    this.videoCanvas = document.getElementById("videoCanvas");
    this.videoCtx = this.videoCanvas.getContext("2d");
    this.writer = new AirWriter(document.getElementById("writingCanvas"));
    this.camera = new BrowserCamera(this.video);
    this.gestures = new GestureRecognizer();
    this.ocr = new BrowserOCR(document.getElementById("ocrOutput"));
    this.mode = "gesture";
    this.hands = null;
    this.running = false;
    this.lastFrameAt = performance.now();
    this.fpsSamples = [];

    this.ui = {
      cameraStatus: document.getElementById("cameraStatus"),
      modelStatus: document.getElementById("modelStatus"),
      fpsStatus: document.getElementById("fpsStatus"),
      gestureLabel: document.getElementById("gestureLabel"),
      motionLabel: document.getElementById("motionLabel"),
      handLabel: document.getElementById("handLabel"),
      toolLabel: document.getElementById("toolLabel"),
      inputLabel: document.getElementById("inputLabel"),
      brushSize: document.getElementById("brushSize"),
      brushSizeLabel: document.getElementById("brushSizeLabel")
    };
  }

  async init() {
    this.bindControls();
    await this.gestures.load();
    this.ui.modelStatus.textContent = this.gestures.modelStatus;
    this.hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });
    this.hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.5
    });
    this.hands.onResults((results) => this.onHandsResults(results));
    this.setToolLabel();
    if (window.lucide) window.lucide.createIcons();
  }

  bindControls() {
    document.getElementById("startCameraBtn").addEventListener("click", () => this.start());
    document.getElementById("stopCameraBtn").addEventListener("click", () => this.stop());
    document.getElementById("gestureModeBtn").addEventListener("click", () => this.setMode("gesture"));
    document.getElementById("writingModeBtn").addEventListener("click", () => this.setMode("writing"));
    document.getElementById("clearBtn").addEventListener("click", () => this.writer.clear());
    document.getElementById("eraserBtn").addEventListener("click", () => {
      this.writer.setEraser();
      this.activateSwatch(null);
      this.setToolLabel();
    });
    document.getElementById("downloadBtn").addEventListener("click", () => this.download());
    document.getElementById("ocrBtn").addEventListener("click", async () => {
      try {
        await this.ocr.recognize(this.writer.exportCanvas("#ffffff"));
      } catch (error) {
        document.getElementById("ocrOutput").value = error.message;
      }
    });
    document.querySelectorAll(".swatch").forEach((button) => {
      button.addEventListener("click", () => {
        this.writer.setColor(button.dataset.tool);
        this.activateSwatch(button.dataset.tool);
        this.setToolLabel();
      });
    });
    this.ui.brushSize.addEventListener("input", () => {
      this.writer.setBrushSize(this.ui.brushSize.value);
      this.ui.brushSizeLabel.textContent = this.writer.brushSize;
      this.setToolLabel();
    });
  }

  async start() {
    try {
      this.ui.cameraStatus.textContent = "Camera starting";
      await this.camera.start();
      this.running = true;
      this.ui.cameraStatus.textContent = "Camera live";
      this.processFrame();
    } catch (error) {
      this.ui.cameraStatus.textContent = error.message;
    }
  }

  stop() {
    this.running = false;
    this.camera.stop();
    this.videoCtx.clearRect(0, 0, this.videoCanvas.width, this.videoCanvas.height);
    this.ui.cameraStatus.textContent = "Camera stopped";
  }

  async processFrame() {
    if (!this.running || !this.camera.active) return;
    if (this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      await this.hands.send({ image: this.video });
    }
    requestAnimationFrame(() => this.processFrame());
  }

  onHandsResults(results) {
    const width = this.video.videoWidth || 1280;
    const height = this.video.videoHeight || 720;
    this.ensureCanvasSize(width, height);

    this.videoCtx.clearRect(0, 0, width, height);
    this.videoCtx.save();
    this.videoCtx.drawImage(results.image, 0, 0, width, height);
    this.videoCtx.restore();

    const landmarks = results.multiHandLandmarks && results.multiHandLandmarks[0];
    const handedness = results.multiHandedness?.[0]?.label || "Unknown";
    const recognized = this.gestures.recognize(landmarks, width, height, handedness);
    const writingMotion = this.writer.update(recognized, this.mode);

    drawHandOverlay(this.videoCtx, recognized);
    this.updateReadout(recognized, writingMotion);
    this.updateFps();
  }

  ensureCanvasSize(width, height) {
    if (this.videoCanvas.width !== width || this.videoCanvas.height !== height) {
      this.videoCanvas.width = width;
      this.videoCanvas.height = height;
    }
    this.writer.resize(width, height);
  }

  updateReadout(result, writingMotion) {
    this.ui.gestureLabel.textContent = result.gesture.replaceAll("_", " ");
    this.ui.motionLabel.textContent = this.mode === "writing" ? writingMotion : result.motion;
    this.ui.handLabel.textContent = result.handedness;
    this.ui.inputLabel.textContent = result.input;
  }

  updateFps() {
    const now = performance.now();
    const fps = 1000 / Math.max(1, now - this.lastFrameAt);
    this.lastFrameAt = now;
    this.fpsSamples.push(fps);
    if (this.fpsSamples.length > 10) this.fpsSamples.shift();
    const average = this.fpsSamples.reduce((sum, value) => sum + value, 0) / this.fpsSamples.length;
    this.ui.fpsStatus.textContent = `FPS ${Math.round(average)}`;
  }

  setMode(mode) {
    this.mode = mode;
    this.writer.stopStroke();
    document.getElementById("gestureModeBtn").classList.toggle("active", mode === "gesture");
    document.getElementById("writingModeBtn").classList.toggle("active", mode === "writing");
  }

  activateSwatch(color) {
    document.querySelectorAll(".swatch").forEach((button) => {
      button.classList.toggle("active", button.dataset.tool === color);
    });
  }

  setToolLabel() {
    const tool = this.writer.tool === "eraser" ? "Eraser" : `${this.writer.color} pen`;
    this.ui.toolLabel.textContent = `${tool}, ${this.writer.brushSize}px`;
  }

  download() {
    const link = document.createElement("a");
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    link.download = `air-writing-${stamp}.png`;
    link.href = this.writer.exportCanvas("transparent").toDataURL("image/png");
    link.click();
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  const app = new WebHandGestureApp();
  await app.init();
  window.webHandGestureApp = app;
});
