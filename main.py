#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import json
import argparse
import itertools
import mimetypes
import socket
import socketserver
import sys
import threading
import time
import webbrowser
import base64
from functools import partial
from pathlib import Path
from collections import Counter, deque
from datetime import datetime

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc

MODEL_DIR = Path(__file__).resolve().parent / 'model'
UI_MOCKUP_PATH = Path(__file__).resolve().parent / 'ui_mockup.html'
APP_TITLE = 'Hand Gesture & Air Writing System'
MODULE_GESTURE = 'Gesture Recognition'
MODULE_AIR_WRITING = 'Air Writing'
OUTPUT_DIR = Path(__file__).resolve().parent / 'output'
REQUIRED_MODEL_ASSETS = (
    MODEL_DIR / 'keypoint_classifier' / 'keypoint_classifier.tflite',
    MODEL_DIR / 'keypoint_classifier' / 'keypoint_classifier_label.csv',
    MODEL_DIR / 'point_history_classifier' / 'point_history_classifier.tflite',
    MODEL_DIR / 'point_history_classifier' / 'point_history_classifier_label.csv',
)
FINGER_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)
COLOR_PANEL = (34, 23, 16)
COLOR_PANEL_LIGHT = (41, 29, 21)
COLOR_TEXT = (251, 248, 246)
COLOR_MUTED = (195, 180, 170)
COLOR_ACCENT = (255, 140, 79)
COLOR_SUCCESS = (106, 240, 72)
COLOR_WARNING = (255, 216, 40)
COLOR_DARK = (18, 14, 10)
COLOR_LANDMARK_LINE = (255, 218, 82)
COLOR_LANDMARK_DOT = (255, 246, 225)
COLOR_LANDMARK_CENTER = (255, 160, 70)
NEON_COLORS = {
    'blue': (255, 154, 52),
    'red': (72, 72, 255),
    'green': (72, 240, 106),
    'yellow': (72, 216, 255),
    'white': (245, 245, 245),
    'purple': (255, 120, 180),
}
KEYPOINT_LABEL_KEYS = {
    'a': 10,  # Saranghae
    'b': 11,  # You
}
KEYPOINT_LOGGING_LABELS = set(range(10)) | set(KEYPOINT_LABEL_KEYS.values())
POINT_HISTORY_LOGGING_LABELS = set(range(10))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)

    return parser.parse_args()


def main():
    args = get_args()
    window_name = APP_TITLE

    validate_model_assets()

    cap = cv.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {args.device}")

    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, args.width, args.height)

    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    try:
        from model import KeyPointClassifier, PointHistoryClassifier

        keypoint_classifier = KeyPointClassifier(
            model_path=str(MODEL_DIR / 'keypoint_classifier' / 'keypoint_classifier.tflite')
        )
        point_history_classifier = PointHistoryClassifier(
            model_path=str(MODEL_DIR / 'point_history_classifier' / 'point_history_classifier.tflite')
        )

        keypoint_labels = load_csv_labels(
            MODEL_DIR / 'keypoint_classifier' / 'keypoint_classifier_label.csv'
        )
        point_labels = load_csv_labels(
            MODEL_DIR / 'point_history_classifier' / 'point_history_classifier_label.csv'
        )

        fps_calc = CvFpsCalc(buffer_len=10)

        history_length = 16
        point_history = deque(maxlen=history_length)
        finger_history = deque(maxlen=history_length)

        mode = 0

        while True:
            fps = fps_calc.get()
            hand_detected = False
            active_finger_text = ''

            key = cv.waitKey(10)
            if key == 27:
                break

            number, mode = select_mode(key, mode)

            ret, image = cap.read()
            if not ret:
                break

            image = cv.flip(image, 1)
            debug_image = copy.deepcopy(image)

            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks:
                hand_detected = True
                handedness_list = results.multi_handedness or []

                for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = handedness_list[index] if index < len(handedness_list) else None
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    processed_landmarks = pre_process_landmark(landmark_list)
                    processed_history = pre_process_point_history(debug_image, point_history)

                    logging_csv(number, mode, processed_landmarks, processed_history)

                    hand_sign_id = keypoint_classifier(processed_landmarks)

                    if hand_sign_id == 2:
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])

                    finger_gesture_id = 0
                    if len(processed_history) == history_length * 2:
                        finger_gesture_id = point_history_classifier(processed_history)

                    finger_history.append(finger_gesture_id)
                    most_common = Counter(finger_history).most_common()
                    active_finger_text = point_labels[most_common[0][0]]

                    debug_image = draw_bounding_rect(True, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        keypoint_labels[hand_sign_id],
                        active_finger_text,
                    )
            else:
                point_history.append([0, 0])

            debug_image = draw_point_history(debug_image, point_history)
            debug_image = draw_info(debug_image, fps, mode, number, hand_detected, active_finger_text)
            if debug_image.shape[1] != args.width or debug_image.shape[0] != args.height:
                debug_image = resize_to_fit_keep_aspect(debug_image, args.width, args.height)

            cv.imshow(window_name, debug_image)
    finally:
        hands.close()
        cap.release()
        cv.destroyAllWindows()


def validate_model_assets():
    missing_assets = [str(path) for path in REQUIRED_MODEL_ASSETS if not path.exists()]
    if missing_assets:
        missing_list = '\n'.join(f"- {path}" for path in missing_assets)
        raise FileNotFoundError(f"Required model assets are missing:\n{missing_list}")


def load_csv_labels(label_path):
    with open(label_path, encoding='utf-8-sig') as f:
        return [row[0] for row in csv.reader(f) if row]


def classify_keypoint_with_confidence(classifier, landmark_list):
    input_details_tensor_index = classifier.input_details[0]['index']
    classifier.interpreter.set_tensor(
        input_details_tensor_index,
        np.array([landmark_list], dtype=np.float32),
    )
    classifier.interpreter.invoke()

    output_details_tensor_index = classifier.output_details[0]['index']
    result = np.squeeze(classifier.interpreter.get_tensor(output_details_tensor_index))
    result_index = int(np.argmax(result))
    return result_index, float(result[result_index])


def select_mode(key, mode):
    number = -1
    label_key = ''
    if 48 <= key <= 57:
        number = key - 48
    if key >= 0:
        label_key = chr(key & 0xFF).lower()
        number = KEYPOINT_LABEL_KEYS.get(label_key, number)
    if label_key == 'n':
        mode = 0
    if label_key == 'k':
        mode = 1
    if label_key == 'h':
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    h, w = image.shape[0], image.shape[1]
    landmark_array = []

    for lm in landmarks.landmark:
        x = min(int(lm.x * w), w - 1)
        y = min(int(lm.y * h), h - 1)
        landmark_array.append([x, y])

    landmark_array = np.array(landmark_array)
    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    h, w = image.shape[0], image.shape[1]
    landmark_points = []

    for lm in landmarks.landmark:
        x = min(int(lm.x * w), w - 1)
        y = min(int(lm.y * h), h - 1)
        landmark_points.append([x, y])

    return landmark_points


def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)

    base_x, base_y = temp[0]

    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y

    temp = list(itertools.chain.from_iterable(temp))

    max_value = max(map(abs, temp))
    if max_value == 0:
        return [0.0 for _ in temp]

    return [n / max_value for n in temp]


def pre_process_point_history(image, point_history):
    h, w = image.shape[0], image.shape[1]

    temp = copy.deepcopy(point_history)
    if not temp:
        # No history yet (e.g., first detected frame); keep shape consistent.
        maxlen = point_history.maxlen if point_history.maxlen else 0
        return [0.0] * (maxlen * 2)

    base_x, base_y = temp[0]

    for i in range(len(temp)):
        temp[i][0] = (temp[i][0] - base_x) / w
        temp[i][1] = (temp[i][1] - base_y) / h

    return list(itertools.chain.from_iterable(temp))


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 1 and number in KEYPOINT_LOGGING_LABELS:
        with open(MODEL_DIR / 'keypoint_classifier' / 'keypoint.csv', 'a', newline="") as f:
            csv.writer(f).writerow([number, *landmark_list])

    if mode == 2 and number in POINT_HISTORY_LOGGING_LABELS:
        with open(MODEL_DIR / 'point_history_classifier' / 'point_history.csv', 'a', newline="") as f:
            csv.writer(f).writerow([number, *point_history_list])


def clip_box(image, left, top, right, bottom):
    height, width = image.shape[:2]
    return (
        max(0, min(width - 1, left)),
        max(0, min(height - 1, top)),
        max(0, min(width - 1, right)),
        max(0, min(height - 1, bottom)),
    )


def draw_translucent_rect(image, left, top, right, bottom, color, alpha=0.72):
    left, top, right, bottom = clip_box(image, left, top, right, bottom)
    if right <= left or bottom <= top:
        return image

    overlay = image.copy()
    cv.rectangle(overlay, (left, top), (right, bottom), color, -1)
    image[top:bottom, left:right] = cv.addWeighted(
        overlay[top:bottom, left:right],
        alpha,
        image[top:bottom, left:right],
        1 - alpha,
        0,
    )
    return image


def draw_text(image, text, origin, font_scale=0.65, color=COLOR_TEXT, thickness=1):
    cv.putText(
        image,
        text,
        origin,
        cv.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv.LINE_AA,
    )
    return image


def text_size(text, font_scale=0.65, thickness=1):
    return cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)


def draw_chip(image, text, left, top, color=COLOR_ACCENT):
    (text_width, text_height), baseline = text_size(text, 0.55, 1)
    right = left + text_width + 22
    bottom = top + text_height + baseline + 14
    draw_translucent_rect(image, left, top, right, bottom, COLOR_PANEL_LIGHT, alpha=0.78)
    cv.rectangle(image, (left, top), (left + 4, bottom), color, -1)
    draw_text(image, text, (left + 12, bottom - baseline - 5), 0.55, COLOR_TEXT, 1)
    return right, bottom


def draw_module_tab(image, text, left, top, active=False):
    color = COLOR_ACCENT if active else COLOR_PANEL_LIGHT
    text_color = COLOR_TEXT if active else COLOR_MUTED
    (text_width, text_height), baseline = text_size(text, 0.52, 1)
    right = left + text_width + 24
    bottom = top + text_height + baseline + 14
    draw_translucent_rect(image, left, top, right, bottom, color, alpha=0.76 if active else 0.58)
    draw_text(image, text, (left + 12, bottom - baseline - 5), 0.52, text_color, 1)
    return right, bottom


def draw_landmarks(image, landmark_point):
    for start, end in FINGER_CONNECTIONS:
        if start < len(landmark_point) and end < len(landmark_point):
            cv.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), COLOR_DARK, 4, cv.LINE_AA)
            cv.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), COLOR_LANDMARK_LINE, 2, cv.LINE_AA)

    for index, point in enumerate(landmark_point):
        radius = 6 if index in (0, 4, 8, 12, 16, 20) else 4
        cv.circle(image, tuple(point), radius + 2, COLOR_DARK, -1, cv.LINE_AA)
        cv.circle(image, tuple(point), radius, COLOR_LANDMARK_DOT, -1, cv.LINE_AA)
        cv.circle(image, tuple(point), max(2, radius - 3), COLOR_LANDMARK_CENTER, -1, cv.LINE_AA)
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        left, top, right, bottom = brect
        corner = max(18, min(36, (right - left) // 5))
        cv.rectangle(image, (left, top), (right, bottom), COLOR_DARK, 2, cv.LINE_AA)
        for color, thickness in ((COLOR_ACCENT, 4), (COLOR_TEXT, 1)):
            cv.line(image, (left, top), (left + corner, top), color, thickness, cv.LINE_AA)
            cv.line(image, (left, top), (left, top + corner), color, thickness, cv.LINE_AA)
            cv.line(image, (right, top), (right - corner, top), color, thickness, cv.LINE_AA)
            cv.line(image, (right, top), (right, top + corner), color, thickness, cv.LINE_AA)
            cv.line(image, (left, bottom), (left + corner, bottom), color, thickness, cv.LINE_AA)
            cv.line(image, (left, bottom), (left, bottom - corner), color, thickness, cv.LINE_AA)
            cv.line(image, (right, bottom), (right - corner, bottom), color, thickness, cv.LINE_AA)
            cv.line(image, (right, bottom), (right, bottom - corner), color, thickness, cv.LINE_AA)
    return image


def draw_text_block(image, text, origin, font_scale=0.7, thickness=2):
    (text_width, text_height), baseline = text_size(text, font_scale, thickness)
    x, y = origin
    draw_translucent_rect(
        image,
        x - 10,
        y - text_height - 12,
        x + text_width + 10,
        y + baseline + 7,
        COLOR_PANEL,
        alpha=0.76,
    )
    draw_text(image, text, (x, y), font_scale, COLOR_TEXT, thickness)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_text):
    label = 'Unknown'
    if handedness and handedness.classification:
        label = handedness.classification[0].label

    if hand_sign_text:
        label += ': ' + hand_sign_text

    font_scale = 0.58
    thickness = 1
    (text_width, text_height), baseline = text_size(label, font_scale, thickness)
    label_left = max(10, min(brect[0], image.shape[1] - text_width - 28))
    label_bottom = max(42, brect[1] - 8)
    label_top = label_bottom - text_height - baseline - 16
    label_right = label_left + text_width + 24
    draw_translucent_rect(image, label_left, label_top, label_right, label_bottom, COLOR_PANEL, alpha=0.88)
    cv.rectangle(image, (label_left, label_top), (label_left + 5, label_bottom), COLOR_ACCENT, -1)
    draw_text(image, label, (label_left + 14, label_bottom - baseline - 6), font_scale, COLOR_TEXT, thickness)

    return image


def draw_point_history(image, point_history):
    for i, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            radius = max(2, int(2 + i * 0.45))
            alpha_color = (
                min(255, COLOR_WARNING[0] + i * 3),
                max(120, COLOR_WARNING[1] - i * 2),
                COLOR_WARNING[2],
            )
            cv.circle(image, (point[0], point[1]), radius + 2, COLOR_DARK, 1, cv.LINE_AA)
            cv.circle(image, (point[0], point[1]), radius, alpha_color, 2, cv.LINE_AA)
    return image


def draw_info(image, fps, mode, number, hand_detected, finger_text):
    height, width = image.shape[:2]
    draw_translucent_rect(image, 0, 0, width - 1, 76, COLOR_PANEL, alpha=0.70)
    cv.line(image, (0, 76), (width, 76), COLOR_ACCENT, 1, cv.LINE_AA)

    draw_text(image, APP_TITLE, (18, 31), 0.66, COLOR_TEXT, 2)
    draw_text(image, "Live vision interface", (19, 56), 0.46, COLOR_MUTED, 1)

    tabs_left = max(18, width - 410)
    tab_right, _ = draw_module_tab(image, MODULE_GESTURE, tabs_left, 22, active=True)
    draw_module_tab(image, MODULE_AIR_WRITING, tab_right + 8, 22, active=False)

    status_text = "HAND DETECTED" if hand_detected else "SEARCHING"
    status_color = COLOR_SUCCESS if hand_detected else COLOR_WARNING
    chip_right, _ = draw_chip(image, status_text, 18, 92, status_color)
    draw_chip(image, f"FPS {fps}", chip_right + 10, 92, COLOR_ACCENT)

    if mode == 1:
        mode_text = "KEYPOINT LOGGING"
    elif mode == 2:
        mode_text = "HISTORY LOGGING"
    else:
        mode_text = "LIVE"

    draw_chip(image, mode_text, 18, 132, COLOR_ACCENT)

    if number in KEYPOINT_LOGGING_LABELS:
        draw_chip(image, f"LABEL {number}", 18, 172, COLOR_WARNING)

    if finger_text:
        draw_chip(image, f"MOTION {finger_text.upper()}", 18, height - 52, COLOR_SUCCESS)
    elif not hand_detected:
        (text_width, text_height), baseline = text_size("No hand in frame", 0.72, 2)
        left = (width - text_width - 34) // 2
        top = height - 82
        draw_translucent_rect(image, left, top, left + text_width + 34, top + text_height + baseline + 22, COLOR_PANEL, 0.72)
        draw_text(image, "No hand in frame", (left + 17, top + text_height + 8), 0.72, COLOR_MUTED, 2)

    return image


def resize_to_fit_keep_aspect(image, target_width, target_height):
    src_h, src_w = image.shape[:2]
    if src_h == 0 or src_w == 0:
        return image

    # Scale to fit target size (no stretch), then pad to keep full frame visible.
    scale = min(target_width / src_w, target_height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_LINEAR)

    canvas = np.full((target_height, target_width, 3), 245, dtype=np.uint8)
    x = (target_width - new_w) // 2
    y = (target_height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def resize_to_cover_crop(image, target_width, target_height):
    src_h, src_w = image.shape[:2]
    if src_h == 0 or src_w == 0:
        return image

    scale = max(target_width / src_w, target_height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    x = max(0, (new_w - target_width) // 2)
    y = max(0, (new_h - target_height) // 2)
    return resized[y:y + target_height, x:x + target_width]


def draw_canvas(canvas, start_point, end_point, color_name='green', brush_size=5, erase=False):
    if canvas is None or start_point is None or end_point is None:
        return
    if len(canvas.shape) == 3 and canvas.shape[2] == 4:
        color = (0, 0, 0, 0) if erase else (*NEON_COLORS.get(color_name, NEON_COLORS['green']), 255)
    else:
        color = (0, 0, 0) if erase else NEON_COLORS.get(color_name, NEON_COLORS['green'])
    cv.line(canvas, tuple(start_point), tuple(end_point), color, int(max(2, brush_size)), cv.LINE_AA)


def detect_color_pointer(frame_bgr):
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    lower = np.array([64, 72, 49], dtype=np.uint8)
    upper = np.array([153, 255, 255], dtype=np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.inRange(hsv, lower, upper)
    mask = cv.erode(mask, kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.dilate(mask, kernel, iterations=1)
    contours_result = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours_result) == 3:
        _, contours, _ = contours_result
    else:
        contours, _ = contours_result
    if not contours:
        return None
    contour = max(contours, key=cv.contourArea)
    if cv.contourArea(contour) < 80:
        return None
    moments = cv.moments(contour)
    if moments['m00'] == 0:
        return None
    return int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])


def handle_gesture(gesture, brush_size):
    action = {
        'draw': gesture == 'Pointer',
        'erase': False,
        'clear': gesture == 'Open',
        'brush_size': brush_size,
    }
    if gesture == 'Thumb_Up':
        action['brush_size'] = min(30, brush_size + 1)
    elif gesture == 'Thumb_Down':
        action['brush_size'] = max(2, brush_size - 1)
    return action


def writing_mode(gesture, brush_size):
    return handle_gesture(gesture, brush_size)


def save_output(canvas_image):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = OUTPUT_DIR / f'drawing_{timestamp}.png'
    cv.imwrite(str(file_path), canvas_image)
    return str(file_path)


def recognize_text_from_canvas(canvas_image):
    try:
        import pytesseract
    except Exception:
        return 'OCR unavailable (install pytesseract + Tesseract).'

    gray = cv.cvtColor(canvas_image, cv.COLOR_BGRA2GRAY if canvas_image.shape[2] == 4 else cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 40, 255, cv.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, config='--psm 6').strip()
    return text or 'No text detected.'


class VisionBackend:
    def __init__(self, device=0, width=960, height=540):
        self.device = device
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.frame = None
        self.canvas_frame = None
        self.mode = 'gesture'
        self.logging_mode = 0
        self.selected_number = -1
        self.selected_color = 'green'
        self.selected_tool = 'pen'
        self.last_color_change_at = 0.0
        self.last_tool_change_at = 0.0
        self.clear_writing_requested = False
        self.state = {
            'camera_on': False,
            'hand_detected': False,
            'gesture': 'No hand',
            'handedness': 'Unknown',
            'motion': 'Stop',
            'confidence': 0.0,
            'fps': 0.0,
            'landmarks': 0,
            'index_point': None,
            'mode': 0,
            'mode_text': 'LIVE',
            'selected_label': -1,
            'module': 'gesture',
            'erase_signal': False,
            'brush_size': 6,
            'brush_label': 'Brush: 6 px',
            'selected_color': 'green',
            'selected_tool': 'pen',
            'status_message': '',
            'updated_at': time.time(),
        }

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        self._set_state(
            camera_on=False,
            hand_detected=False,
            gesture='Camera off',
            handedness='Unknown',
            motion='Stop',
            confidence=0.0,
            fps=0.0,
            landmarks=0,
            index_point=None,
        )
        with self.lock:
            self.frame = None

    def get_frame(self):
        with self.lock:
            return self.frame

    def get_canvas_frame(self):
        with self.lock:
            return self.canvas_frame

    def get_state(self):
        with self.lock:
            return dict(self.state)

    def handle_keyboard_key(self, key):
        key = (key or '').lower()
        with self.lock:
            if len(key) == 1 and key.isdigit():
                self.selected_number = int(key)
            elif key in KEYPOINT_LABEL_KEYS:
                self.selected_number = KEYPOINT_LABEL_KEYS[key]
            elif key == 'k':
                self.logging_mode = 1
            elif key == 'h':
                self.logging_mode = 2
            elif key == 'n':
                self.logging_mode = 0
                self.selected_number = -1
            else:
                return dict(self.state)

            mode_text = self._mode_text(self.logging_mode)
            self.state.update(
                mode=self.logging_mode,
                mode_text=mode_text,
                selected_label=self.selected_number,
                updated_at=time.time(),
            )
            return dict(self.state)

    def set_writing_tool(self, tool):
        tool = (tool or '').lower()
        with self.lock:
            if tool in NEON_COLORS:
                self.selected_color = tool
                self.selected_tool = 'pen'
            elif tool == 'eraser':
                self.selected_tool = 'eraser'
            elif tool == 'pen':
                self.selected_tool = 'pen'
            else:
                return dict(self.state)
        self._set_state(
            selected_color=self.selected_color,
            selected_tool=self.selected_tool,
            status_message=f'Tool: {tool.upper()}',
            motion='Ready',
        )
        return self.get_state()

    def get_logging_config(self):
        with self.lock:
            return self.logging_mode, self.selected_number

    def set_processing_module(self, module):
        if module not in ('gesture', 'writing'):
            return dict(self.state)
        with self.lock:
            self.mode = module
        self._set_state(module=module, erase_signal=False)
        return self.get_state()

    def get_processing_module(self):
        with self.lock:
            return self.mode

    def clear_writing_canvas(self):
        with self.lock:
            self.clear_writing_requested = True
        self._set_state(status_message='Clear', motion='Clear')
        return self.get_state()

    def consume_clear_writing_request(self):
        with self.lock:
            requested = self.clear_writing_requested
            self.clear_writing_requested = False
            return requested

    @staticmethod
    def _mode_text(mode):
        if mode == 1:
            return 'KEYPOINT LOGGING'
        if mode == 2:
            return 'HISTORY LOGGING'
        return 'LIVE'

    def _set_state(self, **kwargs):
        with self.lock:
            kwargs.setdefault('mode', self.logging_mode)
            kwargs.setdefault('mode_text', self._mode_text(self.logging_mode))
            kwargs.setdefault('selected_label', self.selected_number)
            kwargs.setdefault('module', self.mode)
            kwargs.setdefault('selected_color', self.selected_color)
            kwargs.setdefault('selected_tool', self.selected_tool)
            self.state.update(kwargs)
            self.state['updated_at'] = time.time()

    @staticmethod
    def _is_fist(gesture):
        return gesture in ('Close', 'Fist', 'Closed_Fist')

    @staticmethod
    def _is_open_hand(gesture):
        return gesture in ('Open', 'Open_Palm', 'Open_Hand')

    @staticmethod
    def _is_victory(gesture):
        return gesture in ('Victory', 'Victory_Peace')

    @staticmethod
    def _is_pointer(gesture):
        return gesture in ('Pointer', 'Point', 'Index', 'Index_Up')

    def _is_call_me_pose(self, landmark_list, hand_label):
        thumb_up, index_up, middle_up, ring_up, pinky_up = self._finger_states(landmark_list, hand_label)
        return thumb_up and pinky_up and not index_up and not middle_up and not ring_up

    def _gesture_from_fingers(self, model_gesture, landmark_list, hand_label):
        thumb_up, index_up, middle_up, ring_up, pinky_up = self._finger_states(landmark_list, hand_label)
        if thumb_up and pinky_up and not index_up and not middle_up and not ring_up:
            return 'Call_Me'
        if index_up and middle_up and ring_up and pinky_up:
            return 'Open'
        if index_up and middle_up and not ring_up and not pinky_up:
            return 'Victory_Peace'
        if not index_up and not middle_up and not ring_up and not pinky_up:
            return 'Close'
        if index_up:
            return 'Pointer'
        return model_gesture or 'Stop'

    def _draw_writing_toolbar(self, image, draw=True):
        items = [
            ('blue', 'BLUE', NEON_COLORS['blue']),
            ('green', 'GREEN', NEON_COLORS['green']),
            ('red', 'RED', NEON_COLORS['red']),
            ('eraser', 'ERASER', (230, 230, 230)),
            ('clear', 'CLEAR', COLOR_ACCENT),
        ]
        frame_h, frame_w = image.shape[:2]
        gap = 12
        margin = 32
        width = max(112, int((frame_w - (margin * 2) - (gap * (len(items) - 1))) / len(items)))
        height = 58
        left = margin
        top = 82
        regions = []
        for index, (key, label, color) in enumerate(items):
            x1 = left + index * (width + gap)
            x2 = x1 + width
            active = (
                key == self.selected_color
                if key in NEON_COLORS
                else key == self.selected_tool
            )
            if draw:
                panel_color = color if active else COLOR_PANEL
                draw_translucent_rect(image, x1, top, x2, top + height, panel_color, alpha=0.92 if active else 0.86)
                cv.rectangle(image, (x1, top), (x2, top + height), color, 4 if active else 2, cv.LINE_AA)
                (text_width, text_height), baseline = text_size(label, 0.74, 2)
                text_x = x1 + max(8, (width - text_width) // 2)
                text_y = top + (height + text_height) // 2 - baseline + 5
                draw_text(image, label, (text_x + 2, text_y + 2), 0.74, COLOR_DARK, 3)
                draw_text(image, label, (text_x, text_y), 0.74, COLOR_TEXT, 2)
            regions.append((key, x1, top, x2, top + height))
        return regions

    def _select_writing_toolbar(self, point, regions, canvas):
        now = time.time()
        if now - self.last_tool_change_at < 0.65:
            return None
        x, y = point
        for key, x1, y1, x2, y2 in regions:
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.last_tool_change_at = now
                if key == 'clear':
                    return None
                if key == 'eraser':
                    self.selected_tool = 'eraser'
                    return 'Eraser selected'
                self.selected_color = key
                self.selected_tool = 'pen'
                return f'Colour Change {key.upper()}'
        return None

    @staticmethod
    def _count_raised_fingers(landmark_list, hand_label):
        if not landmark_list or len(landmark_list) < 21:
            return 0
        fingers = 0
        if hand_label == 'Right':
            thumb_up = landmark_list[4][0] > landmark_list[3][0]
        else:
            thumb_up = landmark_list[4][0] < landmark_list[3][0]
        fingers += 1 if thumb_up else 0
        for tip_id, pip_id in ((8, 6), (12, 10), (16, 14), (20, 18)):
            if landmark_list[tip_id][1] < landmark_list[pip_id][1]:
                fingers += 1
        return fingers

    @staticmethod
    def _finger_states(landmark_list, hand_label):
        if not landmark_list or len(landmark_list) < 21:
            return False, False, False, False, False
        if hand_label == 'Right':
            thumb_up = landmark_list[4][0] > landmark_list[3][0]
        else:
            thumb_up = landmark_list[4][0] < landmark_list[3][0]
        index_up = landmark_list[8][1] < landmark_list[6][1]
        middle_up = landmark_list[12][1] < landmark_list[10][1]
        ring_up = landmark_list[16][1] < landmark_list[14][1]
        pinky_up = landmark_list[20][1] < landmark_list[18][1]
        return thumb_up, index_up, middle_up, ring_up, pinky_up

    def run_gesture_mode(self, debug_image, context):
        hand_detected = context['hand_detected']
        if not hand_detected:
            context['point_history'].append([0, 0])
            return debug_image

        hand_landmarks = context['hand_landmarks']
        handedness = context['handedness']
        landmark_list = context['landmark_list']
        brect = context['brect']
        processed_history = context['processed_history']
        hand_sign_id = context['hand_sign_id']

        if hand_sign_id == 2:
            context['point_history'].append(landmark_list[8])
            context['index_point'] = {
                'x': landmark_list[8][0] / max(1, debug_image.shape[1]),
                'y': landmark_list[8][1] / max(1, debug_image.shape[0]),
            }
        else:
            context['point_history'].append([0, 0])

        finger_gesture_id = 0
        if len(processed_history) == context['history_length'] * 2:
            finger_gesture_id = context['point_history_classifier'](processed_history)
        context['finger_history'].append(finger_gesture_id)
        most_common = Counter(context['finger_history']).most_common()
        context['motion'] = context['point_labels'][most_common[0][0]]

        debug_image = draw_bounding_rect(True, debug_image, brect)
        debug_image = draw_landmarks(debug_image, landmark_list)
        debug_image = draw_info_text(debug_image, brect, handedness, context['gesture'], context['motion'])
        return debug_image

    def run_writing_mode(self, debug_image, context):
        frame_h, frame_w = debug_image.shape[:2]
        canvas_h, canvas_w = context['writing_canvas'].shape[:2]
        if canvas_h != frame_h or canvas_w != frame_w:
            resized_canvas = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
            copy_h = min(frame_h, canvas_h)
            copy_w = min(frame_w, canvas_w)
            resized_canvas[:copy_h, :copy_w] = context['writing_canvas'][:copy_h, :copy_w]
            context['writing_canvas'] = resized_canvas

        if context.get('clear_writing_requested'):
            context['writing_canvas'][:] = 0
            context['writing_prev_point'] = None
            context['status_message'] = 'Clear'

        context['finger_history'].clear()
        toolbar_regions = self._draw_writing_toolbar(debug_image, draw=False)

        hand_detected = context['hand_detected']
        if not hand_detected:
            context['point_history'].append([0, 0])
            context['writing_prev_point'] = None
            context['smoothed_point'] = None
            context['motion'] = 'No hand'
            context['status_message'] = 'No hand'
        else:
            landmark_list = context['landmark_list']
            gesture = context['gesture']
            current_point = tuple(landmark_list[8])
            context['index_point'] = {
                'x': current_point[0] / max(1, debug_image.shape[1]),
                'y': current_point[1] / max(1, debug_image.shape[0]),
            }
            context['point_history'].append(list(current_point) if self._is_pointer(gesture) else [0, 0])

            debug_image = draw_bounding_rect(True, debug_image, context['brect'])
            debug_image = draw_landmarks(debug_image, landmark_list)
            cv.circle(debug_image, current_point, 10, COLOR_SUCCESS, 2, cv.LINE_AA)

            if self._is_open_hand(gesture):
                if time.time() - context['last_clear_at'] > 1.0:
                    context['writing_canvas'][:] = 0
                    context['last_clear_at'] = time.time()
                    context['status_message'] = 'Clear'
                context['writing_prev_point'] = None
                context['smoothed_point'] = None
                context['motion'] = 'Clear'
            elif self._is_victory(gesture):
                toolbar_message = self._select_writing_toolbar(
                    current_point,
                    toolbar_regions,
                    context['writing_canvas'],
                )
                context['writing_prev_point'] = None
                context['smoothed_point'] = None
                context['motion'] = 'Select'
                context['status_message'] = toolbar_message or 'Two fingers on colour button'
            elif self._is_fist(gesture):
                context['writing_prev_point'] = None
                context['smoothed_point'] = None
                context['motion'] = 'Stop'
                context['status_message'] = 'Stop'
            elif self._is_pointer(gesture):
                smooth = context['smoothed_point']
                if smooth is None:
                    smooth = current_point
                smooth = (
                    int((smooth[0] * 0.35) + (current_point[0] * 0.65)),
                    int((smooth[1] * 0.35) + (current_point[1] * 0.65)),
                )
                if context['writing_prev_point'] is not None:
                    draw_canvas(
                        context['writing_canvas'],
                        context['writing_prev_point'],
                        smooth,
                        color_name=self.selected_color,
                        brush_size=context['brush_size'],
                        erase=self.selected_tool == 'eraser',
                    )
                context['writing_prev_point'] = smooth
                context['smoothed_point'] = smooth
                context['motion'] = 'Erasing' if self.selected_tool == 'eraser' else 'Writing'
                context['status_message'] = (
                    'Erased'
                    if self.selected_tool == 'eraser'
                    else f'Writing ({self.selected_color.upper()})'
                )
            else:
                context['writing_prev_point'] = None
                context['smoothed_point'] = None
                context['motion'] = 'Stop'
                context['status_message'] = 'Stop'

        return self._compose_writing_frame(debug_image, context['writing_canvas'])

    @staticmethod
    def _compose_writing_frame(debug_image, canvas):
        alpha = canvas[:, :, 3:4].astype(np.float32) / 255.0
        canvas_bgr = canvas[:, :, :3]
        return (canvas_bgr * alpha + debug_image * (1.0 - alpha)).astype(np.uint8)

    def _set_frame(self, image):
        ok, buffer = cv.imencode('.jpg', image, [int(cv.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            with self.lock:
                self.frame = buffer.tobytes()

    def _set_canvas_frame(self, canvas):
        ok, buffer = cv.imencode('.png', canvas)
        if ok:
            with self.lock:
                self.canvas_frame = buffer.tobytes()

    def _run(self):
        cap = None
        hands = None
        try:
            from model import KeyPointClassifier, PointHistoryClassifier

            validate_model_assets()
            cap = cv.VideoCapture(self.device)
            if not cap.isOpened():
                self._set_state(camera_on=False, gesture='Camera unavailable')
                return

            cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)

            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
            keypoint_classifier = KeyPointClassifier(
                model_path=str(MODEL_DIR / 'keypoint_classifier' / 'keypoint_classifier.tflite')
            )
            point_history_classifier = PointHistoryClassifier(
                model_path=str(MODEL_DIR / 'point_history_classifier' / 'point_history_classifier.tflite')
            )
            keypoint_labels = load_csv_labels(
                MODEL_DIR / 'keypoint_classifier' / 'keypoint_classifier_label.csv'
            )
            point_labels = load_csv_labels(
                MODEL_DIR / 'point_history_classifier' / 'point_history_classifier_label.csv'
            )

            fps_calc = CvFpsCalc(buffer_len=10)
            history_length = 16
            point_history = deque(maxlen=history_length)
            finger_history = deque(maxlen=history_length)
            writing_canvas = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            writing_prev_point = None
            smoothed_point = None
            last_clear_at = 0.0
            brush_size = 9
            last_module = self.get_processing_module()
            self._set_state(camera_on=True)

            while not self.stop_event.is_set():
                fps = fps_calc.get()
                processing_module = self.get_processing_module()
                ret, image = cap.read()
                if not ret:
                    self._set_state(camera_on=False, gesture='Camera read failed')
                    time.sleep(0.1)
                    continue

                image = cv.flip(image, 1)
                if processing_module == 'writing':
                    image = resize_to_cover_crop(image, self.width, self.height)
                else:
                    image = cv.resize(image, (self.width, self.height), interpolation=cv.INTER_LINEAR)
                debug_image = copy.deepcopy(image)
                rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                rgb_image.flags.writeable = False
                results = hands.process(rgb_image)
                rgb_image.flags.writeable = True

                hand_detected = False
                gesture = 'No hand'
                hand_label = 'Unknown'
                motion = 'Stop'
                confidence = 0.0
                landmarks_count = 0
                index_point = None
                erase_signal = False
                if processing_module != last_module:
                    point_history.clear()
                    finger_history.clear()
                    writing_prev_point = None
                    smoothed_point = None
                    last_module = processing_module

                hand_landmarks = None
                handedness = None
                landmark_list = None
                brect = None
                processed_history = pre_process_point_history(debug_image, point_history)

                if results.multi_hand_landmarks:
                    hand_detected = True
                    handedness_list = results.multi_handedness or []
                    hand_landmarks = results.multi_hand_landmarks[0]
                    handedness = handedness_list[0] if handedness_list else None
                    if handedness and handedness.classification:
                        hand_label = handedness.classification[0].label
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    landmarks_count = len(landmark_list)

                    if processing_module == 'writing':
                        hand_sign_id = -1
                        confidence = (
                            handedness.classification[0].score
                            if handedness and handedness.classification
                            else 0.85
                        )
                        gesture = self._gesture_from_fingers('', landmark_list, hand_label)
                    else:
                        processed_landmarks = pre_process_landmark(landmark_list)
                        mode, selected_number = self.get_logging_config()
                        logging_csv(selected_number, mode, processed_landmarks, processed_history)
                        hand_sign_id, confidence = classify_keypoint_with_confidence(
                            keypoint_classifier,
                            processed_landmarks,
                        )
                        gesture = keypoint_labels[hand_sign_id]
                        if self._is_call_me_pose(landmark_list, hand_label):
                            gesture = 'Call_Me'
                            confidence = max(confidence, 0.92)

                context = {
                    'hand_detected': hand_detected,
                    'hand_landmarks': hand_landmarks,
                    'handedness': handedness,
                    'landmark_list': landmark_list,
                    'brect': brect,
                    'processed_history': processed_history,
                    'hand_sign_id': hand_sign_id if hand_detected else -1,
                    'gesture': gesture,
                    'motion': motion,
                    'index_point': index_point,
                    'erase_signal': erase_signal,
                    'brush_size': brush_size,
                    'writing_canvas': writing_canvas,
                    'writing_prev_point': writing_prev_point,
                    'smoothed_point': smoothed_point,
                    'last_clear_at': last_clear_at,
                    'clear_writing_requested': self.consume_clear_writing_request(),
                    'history_length': history_length,
                    'point_history': point_history,
                    'finger_history': finger_history,
                    'point_history_classifier': point_history_classifier,
                    'point_labels': point_labels,
                    'hand_label': hand_label,
                    'status_message': '',
                }

                if processing_module == 'gesture':
                    debug_image = self.run_gesture_mode(debug_image, context)
                elif processing_module == 'writing':
                    debug_image = self.run_writing_mode(debug_image, context)

                motion = context['motion']
                index_point = context['index_point']
                erase_signal = context['erase_signal']
                brush_size = context['brush_size']
                writing_canvas = context['writing_canvas']
                writing_prev_point = context['writing_prev_point']
                smoothed_point = context['smoothed_point']
                last_clear_at = context['last_clear_at']
                gesture = context['gesture']
                state_gesture = 'Writing Mode' if processing_module == 'writing' else gesture

                if debug_image.shape[1] != self.width or debug_image.shape[0] != self.height:
                    debug_image = resize_to_fit_keep_aspect(debug_image, self.width, self.height)

                self._set_state(
                    camera_on=True,
                    hand_detected=hand_detected,
                    gesture=state_gesture,
                    handedness=hand_label if hand_detected else 'Unknown',
                    motion=motion,
                    confidence=round(confidence * 100, 1) if hand_detected else 0.0,
                    fps=fps,
                    landmarks=landmarks_count,
                    index_point=index_point,
                    erase_signal=erase_signal,
                    brush_size=brush_size,
                    brush_label=f'Brush: {brush_size} px',
                    selected_color=self.selected_color,
                    selected_tool=self.selected_tool,
                    status_message=context.get('status_message', ''),
                )
                self._set_frame(debug_image)
                if processing_module == 'writing':
                    self._set_canvas_frame(writing_canvas)
        except Exception as exc:
            self._set_state(camera_on=False, gesture=f'Runtime error: {exc}')
        finally:
            if hands is not None:
                try:
                    hands.close()
                except Exception:
                    pass
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self._set_state(camera_on=False)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class MiniHTTPRequestHandler(socketserver.StreamRequestHandler):
    server_version = 'VisionMiniHTTP/1.0'

    def __init__(self, *args, directory=None, **kwargs):
        self.directory = Path(directory) if directory else Path.cwd()
        self._headers_buffer = []
        self.headers = {}
        self.command = ''
        self.path = '/'
        super().__init__(*args, **kwargs)

    def handle(self):
        request_line = self.rfile.readline(65537).decode('iso-8859-1', errors='replace').strip()
        if not request_line:
            return
        parts = request_line.split()
        if len(parts) < 2:
            return
        self.command = parts[0].upper()
        self.path = parts[1]
        self.headers = self._read_headers()
        if self.command == 'GET':
            self.do_GET()
        elif self.command == 'POST':
            self.do_POST()
        else:
            self.send_response(405)
            self.end_headers()

    def _read_headers(self):
        headers = {}
        while True:
            line = self.rfile.readline(65537).decode('iso-8859-1', errors='replace')
            if line in ('\r\n', '\n', ''):
                break
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip()] = value.strip()
        return headers

    def send_response(self, code, message=None):
        messages = {
            200: 'OK',
            404: 'Not Found',
            405: 'Method Not Allowed',
        }
        reason = message or messages.get(code, 'OK')
        self._headers_buffer = [f'HTTP/1.1 {code} {reason}\r\n'.encode('iso-8859-1')]
        self.send_header('Server', self.server_version)
        self.send_header('Connection', 'close')

    def send_header(self, keyword, value):
        self._headers_buffer.append(f'{keyword}: {value}\r\n'.encode('iso-8859-1'))

    def end_headers(self):
        self._headers_buffer.append(b'\r\n')
        self.wfile.write(b''.join(self._headers_buffer))
        self._headers_buffer = []

    def do_GET(self):
        request_path = self.path.split('?', 1)[0]
        relative = request_path.lstrip('/') or UI_MOCKUP_PATH.name
        target = (self.directory / relative).resolve()
        base = self.directory.resolve()
        if not str(target).startswith(str(base)) or not target.is_file():
            self.send_response(404)
            self.end_headers()
            return

        body = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or 'application/octet-stream'
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.send_response(404)
        self.end_headers()


class GestureUiHandler(MiniHTTPRequestHandler):
    backend = None

    def log_message(self, format, *args):
        return

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_GET(self):
        request_path = self.path.split('?', 1)[0]

        if request_path in ('/', '/ui_mockup.html'):
            self.path = '/ui_mockup.html'
            return super().do_GET()

        if request_path == '/state':
            payload = json.dumps(self.backend.get_state()).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)
            return

        if request_path.startswith('/keyboard/'):
            key = request_path.rsplit('/', 1)[-1]
            payload = json.dumps(self.backend.handle_keyboard_key(key)).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)
            return

        if request_path == '/camera/on':
            self.backend.start()
            payload = json.dumps({'camera_on': True}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)
            return

        if request_path == '/camera/off':
            self.backend.stop()
            payload = json.dumps({'camera_on': False}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)
            return

        if request_path.startswith('/module/'):
            module = request_path.rsplit('/', 1)[-1].lower()
            payload = json.dumps(self.backend.set_processing_module(module)).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)
            return

        if request_path == '/writing/clear':
            payload = json.dumps(self.backend.clear_writing_canvas()).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)
            return

        if request_path.startswith('/writing/tool/'):
            tool = request_path.rsplit('/', 1)[-1]
            payload = json.dumps(self.backend.set_writing_tool(tool)).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)
            return

        if request_path.startswith('/snapshot.jpg'):
            frame = self.backend.get_frame()
            if frame is None:
                image = np.full((540, 960, 3), (8, 11, 17), dtype=np.uint8)
                state = self.backend.get_state()
                message = 'Camera is off' if not state.get('camera_on') else 'Starting camera...'
                cv.putText(
                    image,
                    message,
                    (330, 270),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (245, 248, 250),
                    2,
                    cv.LINE_AA,
                )
                ok, buffer = cv.imencode('.jpg', image, [int(cv.IMWRITE_JPEG_QUALITY), 82])
                frame = buffer.tobytes() if ok else b''

            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(frame)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(frame)
            return

        if request_path.startswith('/writing/canvas.png'):
            frame = self.backend.get_canvas_frame()
            if frame is None:
                image = np.zeros((540, 960, 4), dtype=np.uint8)
                ok, buffer = cv.imencode('.png', image)
                frame = buffer.tobytes() if ok else b''

            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(frame)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(frame)
            return

        if request_path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            last_frame = None
            while not self.backend.stop_event.is_set():
                frame = self.backend.get_frame()
                if frame and frame != last_frame:
                    try:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                        last_frame = frame
                    except (BrokenPipeError, ConnectionResetError):
                        break
                time.sleep(0.03)
            return

        return super().do_GET()

    def do_POST(self):
        request_path = self.path.split('?', 1)[0]
        content_length = int(self.headers.get('Content-Length', '0') or 0)
        raw_body = self.rfile.read(content_length) if content_length > 0 else b'{}'
        try:
            payload = json.loads(raw_body.decode('utf-8'))
        except Exception:
            payload = {}

        if request_path in ('/writing/save', '/writing/ocr'):
            image_data = payload.get('image', '')
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]
            try:
                decoded = base64.b64decode(image_data)
                array = np.frombuffer(decoded, dtype=np.uint8)
                image = cv.imdecode(array, cv.IMREAD_UNCHANGED)
            except Exception:
                image = None

            if image is None:
                response = {'ok': False, 'message': 'Invalid image payload.'}
            elif request_path == '/writing/save':
                saved_path = save_output(image)
                response = {'ok': True, 'path': saved_path}
            else:
                text = recognize_text_from_canvas(image)
                response = {'ok': True, 'text': text}

            body = json.dumps(response).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()


if __name__ == '__main__':
    if '--qt-writing' in sys.argv:
        from pyqt_writing_ui import main as run_pyqt_writing

        sys.argv = [arg for arg in sys.argv if arg != '--qt-writing']
        run_pyqt_writing()

    if not UI_MOCKUP_PATH.exists():
        raise FileNotFoundError(f"UI file not found: {UI_MOCKUP_PATH}")

    project_dir = Path(__file__).resolve().parent
    port = 8000
    for candidate_port in range(8000, 8020):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(('127.0.0.1', candidate_port)) != 0:
                port = candidate_port
                break

    args = get_args()
    backend = VisionBackend(device=args.device, width=args.width, height=args.height)
    backend.start()

    GestureUiHandler.backend = backend
    handler = partial(GestureUiHandler, directory=str(project_dir))
    server = ThreadingHTTPServer(('127.0.0.1', port), handler)
    url = f'http://127.0.0.1:{port}/{UI_MOCKUP_PATH.name}?v={int(time.time())}'
    print(f'Opening UI: {url}')
    print('Keep this terminal running while using the camera UI. Press Ctrl+C to stop.')
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        backend.stop()
        server.server_close()
