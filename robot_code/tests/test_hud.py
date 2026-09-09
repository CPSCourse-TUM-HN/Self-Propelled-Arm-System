import csv
import os
import tempfile
import time
import unittest

import numpy as np

from demo_core import HudEventRecorder, MissionContext, MissionState, TargetType, load_config
from demo_core.hud import EVENT_FIELDS, render_hud, runtime_hud_snapshot


class FakeBase(object):
    def __init__(self):
        self._motion_lock = None
        self._active_motion = {
            "direction": "forward",
            "speed": 0.18,
            "requested_speed": 0.20,
            "label": "approach_can_forward",
            "started_at": time.time(),
        }


class FakeServices(object):
    def __init__(self):
        self.base = FakeBase()


class FakeRuntime(object):
    def __init__(self):
        self.config = load_config()
        self.context = MissionContext()
        self.context.target_type = TargetType.CAN
        self.context.remember_target(TargetType.CAN, {
            "found": True,
            "confidence": 0.85,
            "error_x": 0.04,
            "bbox_height_norm": 0.30,
            "bbox": [20, 30, 80, 150],
            "center_x": 50,
            "center_y": 90,
        })
        self.context.begin_state(MissionState.APPROACHING)
        self.state = MissionState.APPROACHING
        self.services = FakeServices()
        self.pause_requested = False
        self.stop_requested = False
        self.current_action = None
        self.last_transition = {
            "timestamp": time.time(),
            "from_state": "ALIGNING",
            "event": "TARGET_ALIGNED",
            "to_state": "APPROACHING",
            "reason": None,
        }


class HudTest(unittest.TestCase):
    def test_runtime_snapshot_exposes_effective_command_and_transition(self):
        snapshot = runtime_hud_snapshot(
            FakeRuntime(),
            depth_stats={"mean": 0.6, "min": 0.5, "max": 0.7},
            recording=True,
        )
        self.assertEqual(snapshot["subphase"], "approach_can_forward")
        self.assertEqual(snapshot["transition_event"], "TARGET_ALIGNED")
        self.assertAlmostEqual(snapshot["base_requested_speed"], 0.20)
        self.assertAlmostEqual(snapshot["base_effective_speed"], 0.18)
        self.assertTrue(snapshot["camera_available"])
        self.assertTrue(snapshot["depth_available"])

    def test_render_hud_returns_requested_size_without_mutating_source(self):
        frame = np.zeros((180, 240, 3), dtype=np.uint8)
        original = frame.copy()
        snapshot = runtime_hud_snapshot(FakeRuntime(), depth_stats=None)
        rendered = render_hud(frame, snapshot, enabled=True)
        self.assertEqual(rendered.shape, (480, 640, 3))
        np.testing.assert_array_equal(frame, original)
        self.assertGreater(int(rendered.sum()), 0)

    def test_side_docking_snapshot_exposes_raw_control_and_target_yaw_degrees(self):
        runtime = FakeRuntime()
        runtime.state = MissionState.BIN_SIDE_DOCKING
        runtime.context.begin_state(MissionState.BIN_SIDE_DOCKING)
        runtime.context.last_observation.update({
            "raw_yaw_error_rad": np.deg2rad(-8.0),
            "yaw_error_calibrated_rad": np.deg2rad(-2.0),
            "yaw_target_rad": np.deg2rad(-6.0),
        })
        snapshot = runtime_hud_snapshot(runtime, recording=True)
        self.assertAlmostEqual(snapshot["raw_yaw_deg"], -8.0)
        self.assertAlmostEqual(snapshot["yaw_control_error_deg"], -2.0)
        self.assertAlmostEqual(snapshot["yaw_target_deg"], -6.0)
        rendered = render_hud(np.zeros((240, 320, 3), dtype=np.uint8), snapshot)
        self.assertEqual(rendered.shape, (480, 640, 3))

    def test_front_stage_snapshot_uses_detector_raw_yaw_degrees(self):
        runtime = FakeRuntime()
        runtime.context.last_observation["yaw_error_deg"] = -5.5
        snapshot = runtime_hud_snapshot(runtime, recording=True)
        self.assertAlmostEqual(snapshot["raw_yaw_deg"], -5.5)
        self.assertIsNone(snapshot["yaw_control_error_deg"])

    def test_event_recorder_writes_one_row_per_call(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "demo.events.csv")
            recorder = HudEventRecorder(path)
            recorder.write(runtime_hud_snapshot(FakeRuntime()), 1.25)
            recorder.close()
            with open(path, "r", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(tuple(rows[0].keys()), EVENT_FIELDS)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["state"], "APPROACHING")
            self.assertAlmostEqual(float(rows[0]["video_elapsed_seconds"]), 1.25)


if __name__ == "__main__":
    unittest.main()
