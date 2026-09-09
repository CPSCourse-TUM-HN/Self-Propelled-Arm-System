import os
import tempfile
import unittest

import cv2
import numpy as np
import yaml

from demo_core import load_config
from demo_core.camera_geometry import CameraRectifier
from demo_core.depth_vision import DepthCamera
from demo_core.perception import AprilTagBinDetector, CanDetector, DepthSensor


class _FakeDetection(object):
    ClassID = 1
    Confidence = 0.9
    Left = 5
    Top = 6
    Right = 30
    Bottom = 40


class _FakeNet(object):
    def __init__(self):
        self.last_image = None

    def Detect(self, image):
        self.last_image = image
        return [_FakeDetection()]


class CameraGeometryTest(unittest.TestCase):
    def test_rectifier_preserves_size_and_moves_edge_points(self):
        config = load_config()
        rectifier = CameraRectifier(config)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[::20, :, :] = 255
        frame[:, ::20, :] = 255
        corrected = rectifier.rectify(frame)
        self.assertEqual(corrected.shape, frame.shape)
        point = rectifier.rectify_point(0.0, 0.0)
        self.assertGreater(abs(point[0]) + abs(point[1]), 0.05)

    def test_bad_calibration_resolution_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad.yaml")
            with open(path, "w") as stream:
                yaml.safe_dump({
                    "image_width": 640,
                    "image_height": 480,
                    "camera_matrix": [[300, 0, 160], [0, 300, 120], [0, 0, 1]],
                    "dist_coeff": [[0, 0, 0, 0, 0]],
                }, stream)
            config = load_config(overrides={"camera": {"calibration_yaml": path}})
            with self.assertRaisesRegex(RuntimeError, "does not match configured"):
                CameraRectifier(config)

    def test_missing_calibration_fails_explicitly_when_component_is_created(self):
        config = load_config(overrides={
            "camera": {
                "calibration_yaml": "missing-camera-calibration.yaml",
                "depth_rectification_enabled": True,
            },
        })
        with self.assertRaisesRegex(RuntimeError, "camera calibration file is required"):
            DepthSensor(config)

    def test_depth_and_can_rectification_default_off(self):
        config = load_config()
        depth = DepthSensor(config)
        can = CanDetector(config)
        self.assertFalse(depth.depth_rectification_enabled)
        self.assertEqual(depth.depth_frame_space, "raw")
        self.assertFalse(can.rectification_enabled)
        self.assertEqual(can.frame_space, "raw")

        disabled = DepthSensor(load_config(overrides={
            "camera": {"depth_rectification_enabled": False},
        }))
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertIs(disabled.depth_input_frame(frame), frame)

    def test_depth_camera_preprocessing_honors_rectifier_presence(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(320, dtype=np.uint8)
        camera = DepthCamera.__new__(DepthCamera)
        camera.rectifier = CameraRectifier(load_config(overrides={
            "camera": {"depth_rectification_enabled": True},
        }))
        corrected = camera.prepare_frame(frame)
        self.assertEqual(corrected.shape, frame.shape)
        self.assertFalse(np.array_equal(corrected, frame))
        camera.rectifier = None
        self.assertIs(camera.prepare_frame(frame), frame)

    def test_can_rectification_marks_bbox_coordinate_space(self):
        config = load_config(overrides={
            "runtime": {"dry_run": {"camera": False}},
            "detectors": {"can": {"rectification_enabled": True}},
        })
        detector = CanDetector(config)
        detector.net = _FakeNet()
        detector._cuda_from_numpy = lambda image: image
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(320, dtype=np.uint8)
        results = detector.detect_all(frame)
        self.assertEqual(results[0]["frame_space"], "rectified")
        self.assertEqual(detector.display_frame(frame).shape, frame.shape)
        self.assertIsNotNone(detector.net.last_image)

    def test_raw_detection_point_converts_to_depth_space(self):
        depth = DepthSensor(load_config(overrides={
            "camera": {"depth_rectification_enabled": True},
        }))
        x, y = depth.point_in_depth_space(0.0, 0.0, "raw")
        self.assertGreater(abs(x) + abs(y), 0.05)
        unchanged = depth.point_in_depth_space(x, y, "rectified")
        self.assertAlmostEqual(unchanged[0], x)
        self.assertAlmostEqual(unchanged[1], y)

    def test_apriltag_dry_run_explicitly_remains_raw(self):
        detector = AprilTagBinDetector(load_config())
        result = detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertEqual(result["frame_space"], "raw")


if __name__ == "__main__":
    unittest.main()
