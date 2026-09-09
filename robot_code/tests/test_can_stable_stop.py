from __future__ import print_function

import json
import unittest

import numpy as np

from demo_core import MissionContext, MissionEvent, MissionState, TargetType, load_config
from demo_core.navigation import TargetNavigator, evaluate_approach_stop


class FakeBase(object):
    def __init__(self):
        self.active = None
        self.starts = []

    def start_motion(self, direction, speed, label):
        self.active = (direction, float(speed), label)
        self.starts.append(self.active)

    def motion_active(self, label=None):
        return self.active is not None

    def stop(self):
        self.active = None


class FakeDepth(object):
    def __init__(self, raw_depth):
        self.raw_depth = float(raw_depth)
        self.depth_reads = 0

    def read_frame(self):
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def obstacle_detected_frame(self, frame):
        return False

    def observe_lens_center_frame(self, frame):
        self.depth_reads += 1
        return {"mean": self.raw_depth, "min": self.raw_depth, "max": self.raw_depth}


class FakeDetector(object):
    def __init__(self, bbox_height):
        self.bbox_height = float(bbox_height)

    def confidence_threshold(self, tracking=False):
        return 0.2

    def detect(self, frame):
        return {
            "found": True,
            "confidence": 0.9,
            "error_x": 0.0,
            "bbox_height_norm": self.bbox_height,
            "bbox": [120.0, 60.0, 200.0, 180.0],
            "center_x": 160.0,
            "center_y": 120.0,
        }


class CanStableStopTest(unittest.TestCase):
    def test_three_stop_modes_have_expected_boolean_semantics(self):
        thresholds = {"bbox_height_threshold": 0.36, "depth_raw_threshold": 2.05}
        bbox = evaluate_approach_stop(dict(thresholds, mode="bbox_height"), 0.40, 3.0)
        depth = evaluate_approach_stop(dict(thresholds, mode="depth"), 0.20, 2.0)
        both_false = evaluate_approach_stop(
            dict(thresholds, mode="bbox_height_and_depth"), 0.40, 3.0
        )
        both_true = evaluate_approach_stop(
            dict(thresholds, mode="bbox_height_and_depth"), 0.40, 2.0
        )
        self.assertTrue(bbox["reached"])
        self.assertTrue(depth["reached"])
        self.assertFalse(both_false["reached"])
        self.assertTrue(both_true["reached"])

    def navigator(self, mode, bbox_height, raw_depth):
        config = load_config(overrides={
            "navigation": {"can": {"approach": {
                "stop": {
                    "mode": mode,
                    "bbox_height_threshold": 0.36,
                    "depth_raw_threshold": 2.05,
                },
            }}},
        })
        context = MissionContext()
        context.begin_state(MissionState.APPROACHING)
        base = FakeBase()
        depth = FakeDepth(raw_depth)
        detector = FakeDetector(bbox_height)
        navigator = TargetNavigator(config, context, base, depth, detector, detector)
        return navigator, base, depth

    def test_bbox_only_skips_depthnet_stop_sample(self):
        navigator, _, depth = self.navigator("bbox_height", 0.40, 3.0)
        outcome = navigator.approach_step(TargetType.CAN)
        self.assertEqual(outcome.event, MissionEvent.TARGET_REACHED)
        self.assertEqual(depth.depth_reads, 0)

    def test_and_mode_waits_until_both_thresholds_pass(self):
        navigator, base, depth = self.navigator("bbox_height_and_depth", 0.40, 3.0)
        outcome = navigator.approach_step(TargetType.CAN)
        self.assertIsNone(outcome.event)
        self.assertEqual(base.starts[-1][0], "forward")
        self.assertEqual(depth.depth_reads, 1)

    def test_default_uses_visual_bbox_stop(self):
        config = load_config()
        self.assertEqual(config.get("navigation.can.approach.stop.mode"), "bbox_height")
        self.assertAlmostEqual(config.get("navigation.can.approach.stop.bbox_height_threshold"), 0.36)
        self.assertAlmostEqual(config.get("navigation.can.approach.stop.depth_raw_threshold"), 2.05)

    def test_stop_configuration_rejects_invalid_mode_and_thresholds(self):
        with self.assertRaises(ValueError):
            load_config(overrides={
                "navigation": {"can": {"approach": {"stop": {"mode": "either"}}}},
            })
        with self.assertRaises(ValueError):
            load_config(overrides={
                "navigation": {"can": {"approach": {"stop": {
                    "mode": "depth", "depth_raw_threshold": 0.0,
                }}}},
            })

    def test_notebook_is_valid_and_has_no_saved_outputs(self):
        with open("tuning_tools/can_stable_stop.ipynb", "r", encoding="utf-8") as stream:
            notebook = json.load(stream)
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()
