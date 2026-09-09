from __future__ import print_function

import unittest

import numpy as np

from demo_core import MissionContext, MissionEvent, MissionState, TargetType, load_config
from demo_core.navigation import TargetNavigator


class FakeBase(object):
    def __init__(self):
        self.active = None
        self.starts = []
        self.pulses = []
        self.stop_count = 0

    def start_motion(self, direction, speed, label):
        if self.active == (direction, float(speed), label):
            return False
        self.active = (direction, float(speed), label)
        self.starts.append(self.active)
        return True

    def motion_active(self, label=None):
        return self.active is not None and (label is None or self.active[2] == label)

    def pulse(self, direction, speed, seconds, label):
        self.pulses.append((direction, float(speed), float(seconds), label))

    def stop(self):
        self.active = None
        self.stop_count += 1


class FakeDepth(object):
    def read_frame(self):
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def obstacle_detected_frame(self, frame):
        return False


class SequenceDetector(object):
    def __init__(self, observations):
        self.observations = list(observations)

    def detect(self, frame):
        return dict(self.observations.pop(0))

    def confidence_threshold(self, tracking=False):
        return 0.2


def observation(found, error_x=0.0, height=0.1, pnp_z=None):
    result = {
        "found": bool(found),
        "confidence": 0.9 if found else 0.0,
        "error_x": float(error_x),
        "bbox_height_norm": float(height),
        "bbox": [120.0, 60.0, 200.0, 180.0] if found else None,
        "center_x": 160.0 if found else None,
        "center_y": 120.0 if found else None,
    }
    if pnp_z is not None:
        result["pose"] = {"x": 0.0, "y": 0.0, "z": float(pnp_z)}
        result["distance"] = float(pnp_z)
    return result


class ContinuousNavigationTest(unittest.TestCase):
    def navigator(self, state, observations):
        config = load_config()
        context = MissionContext()
        context.begin_state(state)
        base = FakeBase()
        detector = SequenceDetector(observations)
        navigator = TargetNavigator(config, context, base, FakeDepth(), detector, detector)
        return navigator, base

    def test_fast_search_turn_is_continuous_until_detection(self):
        navigator, base = self.navigator(
            MissionState.SEARCHING,
            [observation(False), observation(True)],
        )
        first = navigator.search_step(TargetType.CAN)
        second = navigator.search_step(TargetType.CAN)
        self.assertIsNone(first.event)
        self.assertEqual(second.event, MissionEvent.TARGET_FOUND)
        self.assertEqual(base.starts, [("left", 0.3, "search_can_routine_0_step_0")])
        self.assertEqual(base.pulses, [])
        self.assertGreaterEqual(base.stop_count, 1)

    def test_straight_approach_is_continuous_and_visual_stop_ends_it(self):
        navigator, base = self.navigator(
            MissionState.APPROACHING,
            [observation(True, height=0.2), observation(True, height=0.5)],
        )
        first = navigator.approach_step(TargetType.CAN)
        second = navigator.approach_step(TargetType.CAN)
        self.assertIsNone(first.event)
        self.assertEqual(second.event, MissionEvent.TARGET_REACHED)
        self.assertEqual(base.starts, [("forward", 0.6, "approach_can_forward")])
        self.assertEqual(base.pulses, [])
        self.assertGreaterEqual(base.stop_count, 1)

    def test_slow_lateral_correction_is_continuous_until_visual_stop(self):
        navigator, base = self.navigator(
            MissionState.APPROACHING,
            [
                observation(True, error_x=0.3, height=0.2),
                observation(True, error_x=0.0, height=0.5),
            ],
        )
        first = navigator.approach_step(TargetType.CAN)
        second = navigator.approach_step(TargetType.CAN)
        self.assertIsNone(first.event)
        self.assertEqual(second.event, MissionEvent.TARGET_REACHED)
        self.assertEqual(base.starts, [("right", 0.15, "approach_can_steer")])
        self.assertEqual(base.pulses, [])
        self.assertGreaterEqual(base.stop_count, 1)

    def test_alignment_turn_is_continuous_until_centered(self):
        navigator, base = self.navigator(
            MissionState.ALIGNING,
            [observation(True, error_x=-0.3), observation(True, error_x=0.0)],
        )
        first = navigator.align_step(TargetType.CAN)
        second = navigator.align_step(TargetType.CAN)
        self.assertIsNone(first.event)
        self.assertEqual(second.event, MissionEvent.TARGET_ALIGNED)
        self.assertEqual(base.starts, [("left", 0.15, "align_can")])
        self.assertEqual(base.pulses, [])
        self.assertGreaterEqual(base.stop_count, 1)

    def test_bin_alignment_uses_configured_target_error_offset(self):
        navigator, base = self.navigator(
            MissionState.ALIGNING,
            [
                observation(True, error_x=0.04),
                observation(True, error_x=-0.16),
            ],
        )

        first = navigator.align_step(TargetType.BIN)
        second = navigator.align_step(TargetType.BIN)

        self.assertIsNone(first.event)
        self.assertEqual(second.event, MissionEvent.TARGET_ALIGNED)
        self.assertEqual(base.starts, [("right", 0.15, "align_bin")])
        self.assertAlmostEqual(second.observation["alignment_target_error_x"], -0.16)
        self.assertAlmostEqual(second.observation["alignment_error_x"], 0.0)

    def test_bin_final_verify_inherits_front_alignment_offset(self):
        navigator, _ = self.navigator(
            MissionState.FINAL_VERIFY,
            [observation(True, error_x=-0.16), observation(True, error_x=-0.16)],
        )
        first = navigator.final_verify_step(TargetType.BIN)
        second = navigator.final_verify_step(TargetType.BIN)
        self.assertIsNone(first.event)
        self.assertEqual(second.event, MissionEvent.TARGET_STABLE)
        self.assertAlmostEqual(second.observation["alignment_target_error_x"], -0.16)

    def test_bin_approach_uses_single_pnp_z_stop_threshold(self):
        navigator, base = self.navigator(
            MissionState.APPROACHING,
            [
                observation(True, pnp_z=0.40),
                observation(True, pnp_z=0.36),
            ],
        )

        far = navigator.approach_step(TargetType.BIN)
        accepted = navigator.approach_step(TargetType.BIN)

        self.assertIsNone(far.event)
        self.assertEqual(accepted.event, MissionEvent.TARGET_REACHED)
        self.assertEqual(base.starts, [("forward", 0.6, "approach_bin_forward")])
        self.assertAlmostEqual(accepted.observation["pnp_stop_threshold_m"], 0.36)

    def test_bin_pnp_stop_waits_if_pose_is_unavailable(self):
        navigator, base = self.navigator(
            MissionState.APPROACHING,
            [observation(True)],
        )

        outcome = navigator.approach_step(TargetType.BIN)

        self.assertIsNone(outcome.event)
        self.assertEqual(outcome.reason, "waiting for calibrated bin PnP pose")
        self.assertEqual(base.starts, [])
        self.assertGreaterEqual(base.stop_count, 1)

    def test_can_alignment_holds_active_turn_for_configured_missing_frame_grace(self):
        navigator, base = self.navigator(
            MissionState.ALIGNING,
            [
                observation(True, error_x=-0.3),
                observation(False),
                observation(False),
                observation(False),
            ],
        )

        navigator.align_step(TargetType.CAN)
        navigator.align_step(TargetType.CAN)
        self.assertTrue(base.motion_active("align_can"))
        navigator.align_step(TargetType.CAN)
        self.assertTrue(base.motion_active("align_can"))
        navigator.align_step(TargetType.CAN)
        self.assertFalse(base.motion_active())
        self.assertEqual(base.stop_count, 1)

    def test_can_alignment_missing_frame_grace_never_starts_motion(self):
        navigator, base = self.navigator(
            MissionState.ALIGNING,
            [observation(False), observation(False)],
        )

        navigator.align_step(TargetType.CAN)
        navigator.align_step(TargetType.CAN)

        self.assertFalse(base.motion_active())
        self.assertEqual(base.starts, [])
        self.assertEqual(base.stop_count, 2)


if __name__ == "__main__":
    unittest.main()
