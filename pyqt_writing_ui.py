#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import copy
import sys
import time
from collections import deque
from datetime import datetime

import cv2 as cv
import mediapipe as mp
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from main import (
    MODEL_DIR,
    OUTPUT_DIR,
    calc_bounding_rect,
    calc_landmark_list,
    classify_keypoint_with_confidence,
    draw_bounding_rect,
    draw_landmarks,
    load_csv_labels,
    pre_process_landmark,
    validate_model_assets,
)


INK_COLORS = {
    "blue": (255, 140, 79, 255),
    "green": (72, 240, 106, 255),
    "red": (255, 45, 85, 255),
    "yellow": (72, 216, 255, 255),
}


class WritingVisionWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    drawing_ready = pyqtSignal(QImage)
    state_ready = pyqtSignal(dict)
    error_ready = pyqtSignal(str)

    def __init__(self, device=0, width=960, height=540, parent=None):
        super().__init__(parent)
        self.device = device
        self.width = width
        self.height = height
        self.running = False
        self.canvas = np.zeros((height, width, 4), dtype=np.uint8)
        self.prev_point = None
        self.smooth_point = None
        self.point_history = deque(maxlen=4)
        self.current_color = "green"
        self.brush_size = 8
        self.last_clear_at = 0.0
        self.last_color_at = 0.0
        self.color_order = list(INK_COLORS)

    def stop(self):
        self.running = False
        self.wait(1500)

    def clear_canvas(self):
        self.canvas[:] = 0
        self.prev_point = None
        self.smooth_point = None
        self.point_history.clear()
        self.drawing_ready.emit(self._canvas_qimage())

    def set_color(self, color):
        if color not in INK_COLORS:
            return
        self.current_color = color
        self.prev_point = None
        self.smooth_point = None

    def save_canvas(self, file_path=None):
        if not file_path:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = str(OUTPUT_DIR / f"pyqt_air_writing_{stamp}.png")
        bgra = cv.cvtColor(self.canvas, cv.COLOR_RGBA2BGRA)
        cv.imwrite(file_path, bgra)
        return file_path

    def export_composite(self, file_path=None):
        if not file_path:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = str(OUTPUT_DIR / f"pyqt_air_writing_export_{stamp}.png")
        background = np.full((self.height, self.width, 3), (7, 16, 27), dtype=np.uint8)
        alpha = self.canvas[:, :, 3:4].astype(np.float32) / 255.0
        rgb = cv.cvtColor(self.canvas[:, :, :3], cv.COLOR_RGB2BGR)
        composite = (rgb * alpha + background * (1.0 - alpha)).astype(np.uint8)
        cv.imwrite(file_path, composite)
        return file_path

    def _canvas_qimage(self):
        image = QImage(
            self.canvas.data,
            self.canvas.shape[1],
            self.canvas.shape[0],
            self.canvas.strides[0],
            QImage.Format_RGBA8888,
        )
        return image.copy()

    def _frame_qimage(self, frame_bgr):
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        return image.copy()

    @staticmethod
    def _finger_states(landmarks, hand_label):
        if not landmarks or len(landmarks) < 21:
            return False, False, False, False, False
        if hand_label == "Right":
            thumb_up = landmarks[4][0] > landmarks[3][0]
        else:
            thumb_up = landmarks[4][0] < landmarks[3][0]
        index_up = landmarks[8][1] < landmarks[6][1]
        middle_up = landmarks[12][1] < landmarks[10][1]
        ring_up = landmarks[16][1] < landmarks[14][1]
        pinky_up = landmarks[20][1] < landmarks[18][1]
        return thumb_up, index_up, middle_up, ring_up, pinky_up

    def _gesture_from_landmarks(self, model_gesture, landmarks, hand_label):
        thumb, index, middle, ring, pinky = self._finger_states(landmarks, hand_label)
        raised = sum((thumb, index, middle, ring, pinky))
        if raised >= 4:
            return "Open"
        if index and middle and not ring and not pinky:
            return "Victory_Peace"
        if index and not middle and not ring and not pinky:
            return "Pointer"
        if raised == 0:
            return "Close"
        return model_gesture or "Unknown"

    def _smooth(self, point):
        alpha = 0.2  # strong smoothing

        if self.smooth_point is None:
            self.smooth_point = point
            return point

        sx = int(self.smooth_point[0] * (1 - alpha) + point[0] * alpha)
        sy = int(self.smooth_point[1] * (1 - alpha) + point[1] * alpha)

        self.smooth_point = (sx, sy)
        return self.smooth_point

    def _draw_point(self, point):
        point = self._smooth(point)

        # ignore noise movement (VERY IMPORTANT)
        if self.prev_point is not None:
            dist = abs(point[0] - self.prev_point[0]) + abs(point[1] - self.prev_point[1])
            if dist < 3:
                return

        rgba = INK_COLORS.get(self.current_color, INK_COLORS["green"])
        bgr = (rgba[2], rgba[1], rgba[0], rgba[3])

        if self.prev_point is not None:
            cv.line(
                self.canvas,
                self.prev_point,
                point,
                bgr,
                self.brush_size,
                cv.LINE_AA
            )

        self.prev_point = point

    def _select_next_color(self):
        now = time.time()
        if now - self.last_color_at < 1.0:
            return
        index = self.color_order.index(self.current_color)
        self.current_color = self.color_order[(index + 1) % len(self.color_order)]
        self.last_color_at = now

    def _handle_gesture(self, gesture, index_point):
        motion = "Stop"
        now = time.time()
        if gesture == "Open":
            if now - self.last_clear_at > 1.0:
                self.clear_canvas()
                self.last_clear_at = now
            motion = "Canvas cleared"
            self.prev_point = None
            self.smooth_point = None
        elif gesture == "Victory_Peace":
            motion = "Color buttons only"
            self.prev_point = None
            self.smooth_point = None
        elif gesture == "Close":
            motion = "Drawing stopped"
            self.prev_point = None
            self.smooth_point = None
        elif gesture == "Pointer" and index_point is not None:
            self._draw_point(index_point)
            motion = "Writing"
        else:
            self.prev_point = None
            self.smooth_point = None
        return motion

    def run(self):
        cap = None
        hands = None
        self.running = True
        try:
            from model import KeyPointClassifier

            validate_model_assets()
            keypoint_classifier = KeyPointClassifier(
                model_path=str(MODEL_DIR / "keypoint_classifier" / "keypoint_classifier.tflite")
            )
            keypoint_labels = load_csv_labels(
                MODEL_DIR / "keypoint_classifier" / "keypoint_classifier_label.csv"
            )
            cap = cv.VideoCapture(self.device)
            if not cap.isOpened():
                self.error_ready.emit(f"Could not open camera device {self.device}")
                return
            cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
            hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
            last_time = time.time()
            fps = 0.0

            while self.running:
                ok, frame = cap.read()
                if not ok:
                    self.error_ready.emit("Camera read failed")
                    time.sleep(0.1)
                    continue

                frame = cv.flip(frame, 1)
                frame = cv.resize(frame, (self.width, self.height), interpolation=cv.INTER_LINEAR)
                debug = copy.deepcopy(frame)
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)
                rgb.flags.writeable = True

                hand_detected = False
                gesture = "No hand"
                confidence = 0.0
                landmarks_count = 0
                hand_label = "Unknown"
                motion = "Searching"
                index_point = None

                if results.multi_hand_landmarks:
                    hand_detected = True
                    hand_landmarks = results.multi_hand_landmarks[0]
                    handedness = results.multi_handedness[0] if results.multi_handedness else None
                    if handedness and handedness.classification:
                        hand_label = handedness.classification[0].label

                    landmark_list = calc_landmark_list(debug, hand_landmarks)
                    landmarks_count = len(landmark_list)
                    processed = pre_process_landmark(landmark_list)
                    hand_sign_id, confidence = classify_keypoint_with_confidence(keypoint_classifier, processed)
                    model_gesture = keypoint_labels[hand_sign_id]
                    gesture = self._gesture_from_landmarks(model_gesture, landmark_list, hand_label)
                    index_point = tuple(landmark_list[8])
                    motion = self._handle_gesture(gesture, index_point)

                    brect = calc_bounding_rect(debug, hand_landmarks)
                    debug = draw_bounding_rect(True, debug, brect)
                    debug = draw_landmarks(debug, landmark_list)
                    cv.circle(debug, index_point, 10, (72, 240, 106), 2, cv.LINE_AA)
                else:
                    self.prev_point = None
                    self.smooth_point = None

                now = time.time()
                elapsed = max(0.001, now - last_time)
                fps = (fps * 0.85) + ((1.0 / elapsed) * 0.15)
                last_time = now

                self.frame_ready.emit(self._frame_qimage(debug))
                self.drawing_ready.emit(self._canvas_qimage())
                self.state_ready.emit(
                    {
                        "camera": "ON",
                        "hand": f"{hand_label} hand" if hand_detected else "No hand",
                        "gesture": gesture,
                        "confidence": round(confidence * 100.0, 1) if hand_detected else 0.0,
                        "landmarks": landmarks_count,
                        "motion": motion,
                        "color": self.current_color,
                        "fps": round(fps, 1),
                    }
                )
                self.msleep(5)
        except Exception as exc:
            self.error_ready.emit(str(exc))
        finally:
            if hands is not None:
                hands.close()
            if cap is not None:
                cap.release()
            self.state_ready.emit(
                {
                    "camera": "OFF",
                    "hand": "No hand",
                    "gesture": "Camera off",
                    "confidence": 0.0,
                    "landmarks": 0,
                    "motion": "Stop",
                    "color": self.current_color,
                    "fps": 0.0,
                }
            )


class WritingModeWindow(QMainWindow):
    def __init__(self, device=0, width=960, height=540):
        super().__init__()
        self.setWindowTitle("Hand Gesture & Air Writing System - Writing Mode")
        self.resize(1240, 820)
        self.worker = WritingVisionWorker(device=device, width=width, height=height)
        self.video_label = QLabel("Starting camera...")
        self.canvas_label = QLabel()
        self.status_label = QLabel("Camera: STARTING")
        self.gesture_label = QLabel("No hand")
        self.confidence_label = QLabel("0.0%")
        self.landmark_label = QLabel("0 / 21")
        self.motion_label = QLabel("Stop")
        self.hand_label = QLabel("No hand")
        self.color_label = QLabel("GREEN")
        self.fps_label = QLabel("0.0")
        self._build_ui()
        self._connect_worker()
        self.worker.start()

    def _build_ui(self):
        root = QWidget()
        layout = QGridLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QLabel("Air Writing Mode")
        title.setObjectName("Title")
        subtitle = QLabel("Index finger draws, open palm clears, color changes only from buttons, closed fist stops.")
        subtitle.setObjectName("Subtitle")

        header = QVBoxLayout()
        header.addWidget(title)
        header.addWidget(subtitle)

        feed_frame = QFrame()
        feed_frame.setObjectName("Panel")
        feed_layout = QGridLayout(feed_frame)
        feed_layout.setContentsMargins(12, 12, 12, 12)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(760, 430)
        self.video_label.setScaledContents(True)
        self.canvas_label.setAlignment(Qt.AlignCenter)
        self.canvas_label.setMinimumSize(760, 170)
        self.canvas_label.setScaledContents(True)
        self.canvas_label.setObjectName("DrawingCanvas")
        feed_layout.addWidget(self.video_label, 0, 0)
        feed_layout.addWidget(self.canvas_label, 1, 0)

        metrics = QFrame()
        metrics.setObjectName("Panel")
        metrics_layout = QGridLayout(metrics)
        metric_items = [
            ("Gesture", self.gesture_label),
            ("Hand", self.hand_label),
            ("Confidence", self.confidence_label),
            ("Landmarks", self.landmark_label),
            ("Motion", self.motion_label),
            ("Color", self.color_label),
            ("FPS", self.fps_label),
        ]
        for row, (name, value) in enumerate(metric_items):
            name_label = QLabel(name)
            name_label.setObjectName("MetricName")
            value.setObjectName("MetricValue")
            metrics_layout.addWidget(name_label, row, 0)
            metrics_layout.addWidget(value, row, 1)

        controls = QFrame()
        controls.setObjectName("Panel")
        controls_layout = QVBoxLayout(controls)
        self.clear_button = QPushButton("Clear")
        self.blue_button = QPushButton("Blue")
        self.green_button = QPushButton("Green")
        self.red_button = QPushButton("Red")
        self.save_button = QPushButton("Save Transparent PNG")
        self.export_button = QPushButton("Export Composite")
        for button in (
            self.blue_button,
            self.green_button,
            self.red_button,
            self.clear_button,
            self.save_button,
            self.export_button,
        ):
            button.setCursor(Qt.PointingHandCursor)
            controls_layout.addWidget(button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.status_label)

        side = QVBoxLayout()
        side.addWidget(metrics)
        side.addWidget(controls)

        layout.addLayout(header, 0, 0, 1, 2)
        layout.addWidget(feed_frame, 1, 0)
        layout.addLayout(side, 1, 1)
        layout.setColumnStretch(0, 1)
        self.setCentralWidget(root)
        self.setStyleSheet(self._stylesheet())

    def _connect_worker(self):
        self.worker.frame_ready.connect(self._set_video)
        self.worker.drawing_ready.connect(self._set_canvas)
        self.worker.state_ready.connect(self._set_state)
        self.worker.error_ready.connect(self._set_error)
        self.clear_button.clicked.connect(self.worker.clear_canvas)
        self.blue_button.clicked.connect(lambda: self.worker.set_color("blue"))
        self.green_button.clicked.connect(lambda: self.worker.set_color("green"))
        self.red_button.clicked.connect(lambda: self.worker.set_color("red"))
        self.save_button.clicked.connect(self._save_canvas)
        self.export_button.clicked.connect(self._export_canvas)

    def _set_video(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def _set_canvas(self, image):
        self.canvas_label.setPixmap(QPixmap.fromImage(image))

    def _set_state(self, state):
        self.status_label.setText(f"Camera: {state.get('camera', 'OFF')}")
        self.gesture_label.setText(str(state.get("gesture", "No hand")).replace("_", " "))
        self.hand_label.setText(str(state.get("hand", "No hand")))
        self.confidence_label.setText(f"{float(state.get('confidence', 0.0)):.1f}%")
        self.landmark_label.setText(f"{int(state.get('landmarks', 0))} / 21")
        self.motion_label.setText(str(state.get("motion", "Stop")))
        self.color_label.setText(str(state.get("color", "green")).upper())
        self.fps_label.setText(f"{float(state.get('fps', 0.0)):.1f}")

    def _set_error(self, message):
        self.status_label.setText(f"Error: {message}")

    def _save_canvas(self):
        default = str(OUTPUT_DIR / f"pyqt_air_writing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save transparent drawing", default, "PNG Images (*.png)")
        if file_path:
            saved = self.worker.save_canvas(file_path)
            self.status_label.setText(f"Saved: {saved}")

    def _export_canvas(self):
        default = str(OUTPUT_DIR / f"pyqt_air_writing_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        file_path, _ = QFileDialog.getSaveFileName(self, "Export composite drawing", default, "PNG Images (*.png)")
        if file_path:
            saved = self.worker.export_composite(file_path)
            self.status_label.setText(f"Exported: {saved}")

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

    @staticmethod
    def _stylesheet():
        return """
        QWidget {
            background: #05070b;
            color: #f6f8fb;
            font-family: Segoe UI;
            font-size: 14px;
        }
        #Title {
            color: #48f06a;
            font-size: 34px;
            font-weight: 900;
            letter-spacing: 1px;
        }
        #Subtitle {
            color: #aab4c3;
            font-size: 14px;
        }
        #Panel {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #101722, stop:1 #080d14);
            border: 1px solid rgba(255, 255, 255, 0.11);
        }
        QLabel {
            border: none;
        }
        #DrawingCanvas {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(72, 240, 106, 0.25);
        }
        #MetricName {
            color: #748093;
            font-weight: 700;
            padding: 8px;
        }
        #MetricValue {
            color: #48f06a;
            font-size: 18px;
            font-weight: 900;
            padding: 8px;
        }
        QPushButton {
            min-height: 44px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f8cff, stop:1 #7d4cff);
            border: 1px solid rgba(255, 255, 255, 0.16);
            color: white;
            font-weight: 900;
            padding: 0 16px;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #28d8ff, stop:1 #48f06a);
        }
        """


def parse_args():
    parser = argparse.ArgumentParser(description="PyQt Writing Mode with OpenCV and MediaPipe.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    window = WritingModeWindow(device=args.device, width=args.width, height=args.height)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
