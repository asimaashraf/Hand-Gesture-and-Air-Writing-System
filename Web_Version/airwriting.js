const INK_COLORS = {
  blue: "#349aff",
  red: "#ff4848",
  green: "#48f06a",
  yellow: "#ffd848",
  white: "#f5f5f5",
  purple: "#b478ff"
};

class AirWriter {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { willReadFrequently: true });
    this.color = "green";
    this.tool = "pen";
    this.brushSize = 7;
    this.prevPoint = null;
    this.smoothedPoint = null;
    this.lastClearAt = 0;
  }

  resize(width, height) {
    if (this.canvas.width === width && this.canvas.height === height) return;
    const snapshot = document.createElement("canvas");
    snapshot.width = this.canvas.width || width;
    snapshot.height = this.canvas.height || height;
    snapshot.getContext("2d").drawImage(this.canvas, 0, 0);
    this.canvas.width = width;
    this.canvas.height = height;
    this.ctx.drawImage(snapshot, 0, 0);
  }

  setColor(color) {
    if (!INK_COLORS[color]) return;
    this.color = color;
    this.tool = "pen";
  }

  setEraser() {
    this.tool = "eraser";
  }

  setBrushSize(size) {
    this.brushSize = Math.max(2, Math.min(30, Number(size) || 7));
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.prevPoint = null;
    this.smoothedPoint = null;
  }

  update(result, mode) {
    if (mode !== "writing" || !result || !result.indexPoint) {
      this.stopStroke();
      return "Stop";
    }

    if (result.gesture === "Open") {
      const now = performance.now();
      if (now - this.lastClearAt > 1000) {
        this.clear();
        this.lastClearAt = now;
      }
      return "Clear";
    }

    if (result.gesture === "Close" || result.gesture === "Victory_Peace") {
      this.stopStroke();
      return result.gesture === "Victory_Peace" ? "Select" : "Stop";
    }

    if (result.gesture !== "Pointer") {
      this.stopStroke();
      return "Stop";
    }

    const current = [result.indexPoint.x, result.indexPoint.y];
    const smooth = this.smooth(current);

    if (this.prevPoint) this.drawLine(this.prevPoint, smooth);
    this.prevPoint = smooth;
    return this.tool === "eraser" ? "Erasing" : "Writing";
  }

  smooth(point) {
    if (!this.smoothedPoint) this.smoothedPoint = point;
    this.smoothedPoint = [
      Math.trunc((this.smoothedPoint[0] * 0.35) + (point[0] * 0.65)),
      Math.trunc((this.smoothedPoint[1] * 0.35) + (point[1] * 0.65))
    ];
    return this.smoothedPoint;
  }

  drawLine(from, to) {
    this.ctx.save();
    this.ctx.lineCap = "round";
    this.ctx.lineJoin = "round";
    this.ctx.lineWidth = this.tool === "eraser" ? this.brushSize * 2.5 : this.brushSize;
    this.ctx.globalCompositeOperation = this.tool === "eraser" ? "destination-out" : "source-over";
    this.ctx.strokeStyle = INK_COLORS[this.color] || INK_COLORS.green;
    this.ctx.beginPath();
    this.ctx.moveTo(from[0], from[1]);
    this.ctx.lineTo(to[0], to[1]);
    this.ctx.stroke();
    this.ctx.restore();
  }

  stopStroke() {
    this.prevPoint = null;
    this.smoothedPoint = null;
  }

  exportCanvas(background = "transparent") {
    if (background === "transparent") return this.canvas;
    const out = document.createElement("canvas");
    out.width = this.canvas.width;
    out.height = this.canvas.height;
    const outCtx = out.getContext("2d");
    outCtx.fillStyle = background;
    outCtx.fillRect(0, 0, out.width, out.height);
    outCtx.drawImage(this.canvas, 0, 0);
    return out;
  }
}

window.AirWriter = AirWriter;
window.INK_COLORS = INK_COLORS;
