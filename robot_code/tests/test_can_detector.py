import sys
import types
import unittest
from unittest import mock

import numpy as np

from demo_core import load_config
from demo_core.perception import CanDetector


class FakeDetection(object):
    def __init__(self, class_id, confidence, bbox):
        self.ClassID = class_id
        self.Confidence = confidence
        self.Left, self.Top, self.Right, self.Bottom = bbox


class FakeDetectNet(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.clustering = None
        self.last_image = None

    def SetClusteringThreshold(self, value):
        self.clustering = value

    def Detect(self, image):
        self.last_image = image
        return [
            FakeDetection(0, 0.99, [0, 0, 20, 20]),
            FakeDetection(1, 0.81, [80, 30, 200, 210]),
            FakeDetection(1, 0.72, [210, 70, 280, 200]),
        ]


class CanDetectorTest(unittest.TestCase):
    def test_native_backend_builds_detectnet_and_returns_common_observation(self):
        created = []

        def detect_net(**kwargs):
            net = FakeDetectNet(**kwargs)
            created.append(net)
            return net

        jetson_inference = types.ModuleType("jetson_inference")
        jetson_inference.detectNet = detect_net
        jetson_utils = types.ModuleType("jetson_utils")
        jetson_utils.cudaFromNumpy = lambda image: image
        config = load_config(overrides={
            "runtime": {"dry_run": {"camera": False}},
        })
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :, 0] = 17
        frame[:, :, 2] = 91
        with mock.patch.dict(sys.modules, {
            "jetson_inference": jetson_inference,
            "jetson_utils": jetson_utils,
        }):
            detector = CanDetector(config)
            all_results = detector.detect_all(frame)
            result = detector.detect(frame)

        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].kwargs["input_blob"], "input_0")
        self.assertEqual(created[0].kwargs["output_cvg"], "scores")
        self.assertEqual(created[0].kwargs["output_bbox"], "boxes")
        self.assertAlmostEqual(created[0].kwargs["threshold"], 0.2)
        self.assertAlmostEqual(created[0].clustering, 0.3)
        self.assertEqual(int(created[0].last_image[0, 0, 0]), 91)
        self.assertEqual(int(created[0].last_image[0, 0, 2]), 17)
        self.assertTrue(result["found"])
        self.assertEqual(len(all_results), 2)
        self.assertAlmostEqual(all_results[1]["confidence"], 0.72)
        self.assertEqual(result["backend"], "detectnet_native")
        self.assertEqual(result["class_id"], 1)
        self.assertAlmostEqual(result["confidence"], 0.81)
        self.assertAlmostEqual(result["center_x"], 140.0)
        self.assertAlmostEqual(result["bbox_height_norm"], 0.75)

    def test_detect_all_keeps_candidates_while_detect_returns_best(self):
        config = load_config()
        detector = CanDetector(config)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detections = detector.detect_all(frame)
        self.assertEqual(len(detections), 1)
        self.assertTrue(detections[0]["simulated"])
        self.assertEqual(detector.detect(frame)["confidence"], detections[0]["confidence"])

if __name__ == "__main__":
    unittest.main()
