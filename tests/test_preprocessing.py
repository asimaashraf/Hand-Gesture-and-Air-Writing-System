import unittest
from collections import deque

import numpy as np

from main import (
    calc_bounding_rect,
    calc_landmark_list,
    load_csv_labels,
    pre_process_landmark,
    pre_process_point_history,
    select_mode,
)


class Landmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Landmarks:
    def __init__(self, points):
        self.landmark = [Landmark(x, y) for x, y in points]


class PreprocessingTests(unittest.TestCase):
    def test_select_mode_switches_modes_and_numbers(self):
        number, mode = select_mode(ord('5'), 0)
        self.assertEqual(number, 5)
        self.assertEqual(mode, 0)

        _, mode = select_mode(ord('k'), 0)
        self.assertEqual(mode, 1)

        _, mode = select_mode(ord('h'), 1)
        self.assertEqual(mode, 2)

        _, mode = select_mode(ord('n'), 2)
        self.assertEqual(mode, 0)

        number, mode = select_mode(ord('A'), 0)
        self.assertEqual(number, 10)
        self.assertEqual(mode, 0)

        number, mode = select_mode(ord('B'), 0)
        self.assertEqual(number, 11)
        self.assertEqual(mode, 0)

        _, mode = select_mode(ord('K'), 0)
        self.assertEqual(mode, 1)

    def test_pre_process_landmark_normalizes_relative_to_wrist(self):
        result = pre_process_landmark([[10, 10], [20, 10], [10, 30]])
        self.assertEqual(result, [0.0, 0.0, 0.5, 0.0, 0.0, 1.0])

    def test_pre_process_landmark_handles_zero_input(self):
        result = pre_process_landmark([[5, 5], [5, 5]])
        self.assertEqual(result, [0.0, 0.0, 0.0, 0.0])

    def test_pre_process_point_history_preserves_expected_length_when_empty(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        point_history = deque(maxlen=16)
        result = pre_process_point_history(image, point_history)
        self.assertEqual(len(result), 32)
        self.assertTrue(all(value == 0.0 for value in result))

    def test_pre_process_point_history_normalizes_by_frame_size(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        point_history = deque([[10, 10], [30, 20]], maxlen=16)
        result = pre_process_point_history(image, point_history)
        self.assertEqual(result, [0.0, 0.0, 0.1, 0.1])

    def test_landmark_points_are_clamped_to_image_bounds(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        landmarks = Landmarks([(0.5, 0.5), (1.5, -0.2)])

        points = calc_landmark_list(image, landmarks)

        self.assertEqual(points, [[100, 50], [199, -20]])

    def test_bounding_rect_uses_landmark_pixel_coordinates(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        landmarks = Landmarks([(0.1, 0.2), (0.3, 0.4)])

        self.assertEqual(calc_bounding_rect(image, landmarks), [20, 20, 61, 41])

    def test_load_csv_labels_ignores_blank_rows(self):
        labels = load_csv_labels('model/keypoint_classifier/keypoint_classifier_label.csv')
        self.assertIn('Open', labels)
        self.assertIn('Rock_On', labels)


if __name__ == '__main__':
    unittest.main()
