const FINGER_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17]
];

class RingBuffer {
  constructor(maxLength) {
    this.maxLength = maxLength;
    this.items = [];
  }

  push(value) {
    this.items.push(value);
    while (this.items.length > this.maxLength) this.items.shift();
  }

  clear() {
    this.items = [];
  }

  get length() {
    return this.items.length;
  }
}

class GestureRecognizer {
  constructor() {
    this.keypointLabels = [];
    this.pointLabels = [];
    this.modelStatus = "Loading";
    this.usingModel = false;
    this.pointHistory = new RingBuffer(16);
    this.fingerHistory = new RingBuffer(16);
  }

  async load() {
    const [keyLabels, pointLabels] = await Promise.all([
      this.loadLabels("models/keypoint_classifier_label.csv"),
      this.loadLabels("models/point_history_classifier_label.csv")
    ]);
    this.keypointLabels = keyLabels;
    this.pointLabels = pointLabels;
    this.modelStatus = "Landmark recognizer ready";
  }

  async loadLabels(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Unable to load ${path}`);
    const text = await response.text();
    return text.split(/\r?\n/).map((row) => row.trim()).filter(Boolean);
  }

  recognize(landmarks, width, height, handednessLabel) {
    if (!landmarks || landmarks.length < 21) {
      this.pointHistory.push([0, 0]);
      return {
        gesture: "No hand",
        motion: "Stop",
        landmarkList: [],
        bbox: null,
        indexPoint: null,
        handedness: "None",
        input: this.usingModel ? "Model" : "Fallback"
      };
    }

    const landmarkList = this.calcLandmarkList(landmarks, width, height);
    const processed = this.preProcessLandmark(landmarkList);
    let handSignId = this.classifyKeypointFallback(landmarkList, handednessLabel);

    const modelGesture = this.keypointLabels[handSignId] || "Unknown";
    const gesture = this.refineGesture(modelGesture, landmarkList, handednessLabel);
    const indexPoint = landmarkList[8];

    if (gesture === "Pointer") {
      this.pointHistory.push(indexPoint);
    } else {
      this.pointHistory.push([0, 0]);
    }

    const processedHistory = this.preProcessPointHistory(width, height);
    let motionId = 0;
    if (processedHistory.length === 32) motionId = this.classifyPointHistoryFallback();

    this.fingerHistory.push(motionId);
    const motion = this.pointLabels[this.mode(this.fingerHistory.items)] || "Stop";

    return {
      gesture,
      motion,
      landmarkList,
      bbox: this.calcBoundingRect(landmarkList),
      indexPoint: { x: indexPoint[0], y: indexPoint[1] },
      handedness: handednessLabel || "Unknown",
      input: "Landmarks"
    };
  }

  calcLandmarkList(landmarks, width, height) {
    return landmarks.map((lm) => [
      Math.min(Math.trunc(lm.x * width), width - 1),
      Math.min(Math.trunc(lm.y * height), height - 1)
    ]);
  }

  calcBoundingRect(points) {
    const xs = points.map((point) => point[0]);
    const ys = points.map((point) => point[1]);
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  }

  preProcessLandmark(points) {
    const baseX = points[0][0];
    const baseY = points[0][1];
    const flattened = points.flatMap((point) => [point[0] - baseX, point[1] - baseY]);
    const maxValue = Math.max(...flattened.map((value) => Math.abs(value)));
    if (maxValue === 0) return flattened.map(() => 0);
    return flattened.map((value) => value / maxValue);
  }

  preProcessPointHistory(width, height) {
    if (!this.pointHistory.length) return Array(32).fill(0);
    const padded = [...this.pointHistory.items];
    while (padded.length < 16) padded.unshift([0, 0]);
    const [baseX, baseY] = padded[0];
    return padded.flatMap((point) => [
      (point[0] - baseX) / width,
      (point[1] - baseY) / height
    ]);
  }

  refineGesture(modelGesture, points, handednessLabel) {
    const [thumbUp, indexUp, middleUp, ringUp, pinkyUp] = this.fingerStates(points, handednessLabel);
    if (thumbUp && indexUp && middleUp && ringUp && pinkyUp) return "Open";
    if (indexUp && middleUp && !ringUp && !pinkyUp) return "Victory_Peace";
    if (!indexUp && !middleUp && !ringUp && !pinkyUp) return "Close";
    if (indexUp && !middleUp && !ringUp && !pinkyUp) return "Pointer";
    return modelGesture || "Stop";
  }

  classifyKeypointFallback(points, handednessLabel) {
    const refined = this.refineGesture("", points, handednessLabel);
    const index = this.keypointLabels.indexOf(refined);
    return index >= 0 ? index : 0;
  }

  classifyPointHistoryFallback() {
    const points = this.pointHistory.items.filter((point) => point[0] || point[1]);
    if (points.length < 4) return 0;
    const first = points[0];
    const last = points[points.length - 1];
    const dx = last[0] - first[0];
    const dy = last[1] - first[1];
    const distance = Math.hypot(dx, dy);
    if (distance < 20) return 0;
    return 3;
  }

  fingerStates(points, handednessLabel) {
    const rightHand = handednessLabel === "Right";
    const thumbUp = rightHand ? points[4][0] > points[3][0] : points[4][0] < points[3][0];
    const indexUp = points[8][1] < points[6][1];
    const middleUp = points[12][1] < points[10][1];
    const ringUp = points[16][1] < points[14][1];
    const pinkyUp = points[20][1] < points[18][1];
    return [thumbUp, indexUp, middleUp, ringUp, pinkyUp];
  }

  mode(values) {
    const counts = new Map();
    values.forEach((value) => counts.set(value, (counts.get(value) || 0) + 1));
    let best = values[0] || 0;
    counts.forEach((count, value) => {
      if (count > (counts.get(best) || 0)) best = value;
    });
    return best;
  }
}

function drawHandOverlay(ctx, result) {
  if (!result.landmarkList.length) return;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  FINGER_CONNECTIONS.forEach(([start, end]) => {
    const a = result.landmarkList[start];
    const b = result.landmarkList[end];
    ctx.strokeStyle = "#ffd852";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  });
  result.landmarkList.forEach((point, index) => {
    ctx.fillStyle = index === 8 ? "#48f06a" : "#fff6e1";
    ctx.beginPath();
    ctx.arc(point[0], point[1], index === 8 ? 7 : 5, 0, Math.PI * 2);
    ctx.fill();
  });
  if (result.bbox) {
    const [x1, y1, x2, y2] = result.bbox;
    ctx.strokeStyle = "#ff8c4f";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
  ctx.restore();
}

window.GestureRecognizer = GestureRecognizer;
window.drawHandOverlay = drawHandOverlay;
