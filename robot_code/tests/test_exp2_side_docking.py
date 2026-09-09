import time
import unittest

import numpy as np

from demo_core import MissionContext, MissionEvent, load_config
from demo_core.navigation import BinSideDockingNavigator
from demo_core.perception import AprilTagBinDetector, tag_size_metrics, tag_yaw_from_rotation_matrix


class FakeBase(object):
    def __init__(self):
        self.pulses = []
        self.continuous = []
        self.stop_count = 0

    def pulse(self, direction, speed, seconds, label):
        self.pulses.append((direction, float(speed), float(seconds), label))

    def start_motion(self, direction, speed, label):
        self.continuous.append((direction, float(speed), label))

    def stop(self):
        self.stop_count += 1


class FakeDepth(object):
    def __init__(self):
        self.read_count = 0

    def read_frame(self):
        self.read_count += 1
        return np.zeros((240, 320, 3), dtype=np.uint8)


class SequenceDetector(object):
    def __init__(self, observations):
        self.observations = list(observations)

    def detect(self, frame):
        return self.observations.pop(0)


def observation(error_x=0.0, yaw=0.0, corner=0.0, height=0.15):
    return {
        "found": True,
        "error_x": error_x,
        "yaw_error_rad": yaw,
        "yaw_source": "test",
        "max_corner_angle_error_deg": corner,
        "bbox_height_norm": height,
    }


class Exp2SideDockingTest(unittest.TestCase):
    def test_rotation_matrix_recovers_signed_left_right_yaw(self):
        for expected in (0.25, -0.25):
            cosine = np.cos(expected)
            sine = np.sin(expected)
            yaw_rotation = np.array([
                [cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine],
            ])
            face_camera = np.diag([1.0, -1.0, -1.0])
            rotation = np.dot(yaw_rotation, face_camera)
            actual = tag_yaw_from_rotation_matrix(rotation)
            self.assertAlmostEqual(actual, expected, places=3)

    def settings(self, **overrides):
        settings = dict(load_config(overrides={
            "navigation": {"bin": {"side_docking": {"experimental": {"enabled": True}}}},
        }).get("navigation.bin.side_docking.experimental"))
        settings.update({
            "post_correction_camera_settle_seconds": 0.0,
            "yaw_positive_turn_direction": "left",
            "correction_mode": "continuous",
            "target_yaw_rad": 0.0,
            "coarse_yaw_center_guard_norm": 0.13,
            "yaw_tolerance_rad": 0.0698131701,
        })
        settings.update(overrides)
        return settings

    def runtime_parts(self, observations, config_overrides=None):
        context = MissionContext()
        context.begin_state(type("State", (), {"value": "BIN_SIDE_DOCKING"})())
        base = FakeBase()
        navigator = BinSideDockingNavigator(
            load_config(overrides=config_overrides),
            context, base, FakeDepth(), SequenceDetector(observations)
        )
        return context, base, navigator

    def test_missing_pose_has_no_yaw_fallback(self):
        config = load_config()
        detector = AprilTagBinDetector(config)
        points = np.array([[120, 80], [200, 80], [200, 160], [120, 160]], dtype=np.float32)
        result = detector._estimate_yaw(points, 320, 240, pose_result=None)
        self.assertIsNone(result["yaw_error_rad"])
        self.assertIsNone(result["yaw_source"])

    def test_missing_calibration_file_fails_explicitly(self):
        detector = AprilTagBinDetector(load_config(overrides={
            "camera": {"calibration_yaml": "missing-calibration.yaml"},
        }))
        with self.assertRaisesRegex(RuntimeError, "calibration file is required"):
            detector._load_calibration()

    def test_tag_geometry_reports_square_and_perspective_errors(self):
        square = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32)
        metrics = tag_size_metrics(square, 100, 100)
        self.assertAlmostEqual(metrics["vertical_edge_error"], 0.0)
        self.assertAlmostEqual(metrics["horizontal_edge_error"], 0.0)
        self.assertAlmostEqual(metrics["max_corner_angle_error_deg"], 0.0)

        perspective = np.array([[10, 8], [30, 12], [30, 28], [10, 32]], dtype=np.float32)
        metrics = tag_size_metrics(perspective, 100, 100)
        self.assertLess(metrics["vertical_edge_error"], 0.0)

    def test_x_guard_preempts_yaw_when_both_axes_are_out(self):
        context, base, navigator = self.runtime_parts([observation(error_x=0.2, yaw=0.2)])
        outcome = navigator.step(self.settings())
        self.assertIsNone(outcome.event)
        self.assertEqual(outcome.reason, "exp2 x guard preempted yaw")
        self.assertEqual(context.state_data["exp2_next_axis"], "center")
        self.assertEqual(base.continuous, [])
        self.assertEqual(base.pulses, [])

    def test_negative_yaw_uses_separate_right_calibration(self):
        yaw = -0.1
        _, base, navigator = self.runtime_parts([observation(yaw=yaw)])
        navigator.step(self.settings())
        self.assertEqual(base.continuous[0], ("right", 0.15, "exp2_yaw_align"))

    def test_pulse_mode_uses_configured_yaw_and_center_durations(self):
        context, base, navigator = self.runtime_parts([
            observation(error_x=0.1, yaw=0.2),
            observation(error_x=0.2, yaw=0.1),
        ])
        settings = self.settings(
            correction_mode="pulse",
            center_stage_entry_yaw_rad=0.1745329252,
            yaw_correction_seconds=0.13,
            center_correction_seconds=0.17,
        )
        navigator.step(settings)
        navigator.step(settings)
        self.assertEqual(base.continuous, [])
        self.assertEqual(
            base.pulses,
            [
                ("left", 0.15, 0.13, "exp2_yaw_align"),
                ("backward", 0.2, 0.17, "exp2_center_align"),
            ],
        )
        self.assertEqual(context.state_data["exp2_pulse_stage"], "center")

    def test_pulse_mode_finishes_coarse_yaw_before_center_stage(self):
        _, base, navigator = self.runtime_parts([
            observation(error_x=0.1, yaw=0.3),
            observation(error_x=0.1, yaw=0.2),
            observation(error_x=0.2, yaw=0.1),
        ])
        settings = self.settings(
            correction_mode="pulse",
            center_stage_entry_yaw_rad=0.1745329252,
            coarse_yaw_center_guard_norm=0.13,
        )
        for _ in range(3):
            navigator.step(settings)
        self.assertEqual(
            [pulse[3] for pulse in base.pulses],
            ["exp2_yaw_align", "exp2_yaw_align", "exp2_center_align"],
        )

    def test_coarse_yaw_center_guard_recenters_before_resuming_yaw(self):
        context, base, navigator = self.runtime_parts([
            observation(error_x=0.14, yaw=0.5),
            observation(error_x=0.12, yaw=0.5),
        ])
        settings = self.settings(
            correction_mode="pulse",
            center_stage_entry_yaw_rad=0.1745329252,
            coarse_yaw_center_guard_norm=0.13,
        )
        navigator.step(settings)
        navigator.step(settings)
        self.assertEqual(
            [pulse[3] for pulse in base.pulses],
            ["exp2_center_align", "exp2_yaw_align"],
        )
        self.assertEqual(context.state_data["exp2_pulse_stage"], "coarse_yaw")

    def test_pulse_center_stage_is_latched_until_center_is_accurate(self):
        context, base, navigator = self.runtime_parts([
            observation(error_x=0.2, yaw=0.1),
            observation(error_x=0.2, yaw=0.3),
            observation(error_x=0.0, yaw=0.3),
        ])
        settings = self.settings(correction_mode="pulse")
        for _ in range(3):
            navigator.step(settings)
        self.assertEqual(
            [pulse[3] for pulse in base.pulses],
            ["exp2_center_align", "exp2_center_align", "exp2_yaw_align"],
        )
        self.assertEqual(context.state_data["exp2_pulse_stage"], "final_yaw")

    def test_requires_multiple_stable_frames(self):
        context, _, navigator = self.runtime_parts([
            observation(corner=180.0, height=1.0),
            observation(corner=180.0, height=1.0),
        ])
        settings = dict(self.settings())
        settings["stable_frames"] = 2
        first = navigator.step(settings)
        second = navigator.step(settings)
        self.assertIsNone(first.event)
        self.assertEqual(second.event, MissionEvent.TARGET_STABLE)
        self.assertEqual(context.state_data["stable_frames"], 2)

    def test_completion_requires_center_and_yaw_within_tolerance_together(self):
        context, _, navigator = self.runtime_parts([
            observation(error_x=0.2, yaw=0.0),
            observation(error_x=0.2, yaw=0.0),
            observation(error_x=0.0, yaw=0.1),
            observation(error_x=0.0, yaw=0.1),
            observation(error_x=0.0, yaw=0.0),
            observation(error_x=0.0, yaw=0.0),
        ])
        settings = self.settings(stable_frames=1)
        outcomes = [navigator.step(settings) for _ in range(6)]
        self.assertTrue(all(outcome.event is None for outcome in outcomes[:-1]))
        self.assertEqual(outcomes[-1].event, MissionEvent.TARGET_STABLE)
        self.assertEqual(context.state_data["stable_frames"], 1)

    def test_center_correction_uses_continuous_linear_motion(self):
        context, base, navigator = self.runtime_parts([observation(error_x=0.2)])
        context.state_data["exp2_next_axis"] = "center"
        outcome = navigator.step(self.settings())
        self.assertIsNone(outcome.event)
        self.assertEqual(base.continuous, [("backward", 0.2, "exp2_center_align")])
        self.assertEqual(base.pulses, [])

    def test_left_side_tag_uses_forward_correction_after_mapping_flip(self):
        context, base, navigator = self.runtime_parts([observation(error_x=-0.2)])
        context.state_data["exp2_next_axis"] = "center"
        outcome = navigator.step(self.settings())
        self.assertIsNone(outcome.event)
        self.assertEqual(base.continuous, [("forward", 0.2, "exp2_center_align")])

    def test_configured_target_yaw_defines_zero_control_error(self):
        target_yaw = np.deg2rad(-6.0)
        _, base, navigator = self.runtime_parts([observation(yaw=target_yaw)])
        outcome = navigator.step(self.settings(target_yaw_rad=target_yaw))
        self.assertIsNone(outcome.event)
        self.assertEqual(base.pulses, [])
        self.assertEqual(base.continuous, [])
        self.assertAlmostEqual(outcome.observation["yaw_error_calibrated_rad"], 0.0)
        self.assertAlmostEqual(outcome.observation["yaw_target_deg"], -6.0)

    def test_x_guard_starts_center_correction_before_yaw(self):
        frames = [
            observation(error_x=0.2, yaw=1.0),
            observation(error_x=0.2, yaw=0.0),
            observation(error_x=0.2, yaw=0.0),
        ]
        _, base, navigator = self.runtime_parts(frames)
        settings = self.settings()
        for _ in frames:
            navigator.step(settings)
        self.assertEqual(
            [motion[2] for motion in base.continuous],
            ["exp2_center_align", "exp2_center_align"],
        )

    def test_large_final_yaw_is_corrected(self):
        frames = [observation(yaw=1.0)]
        _, base, navigator = self.runtime_parts(frames)
        settings = self.settings()
        navigator.step(settings)
        self.assertEqual(base.continuous[0], ("left", 0.15, "exp2_yaw_align"))

    def test_yaw_correction_settle_period_skips_camera_reads(self):
        context = MissionContext()
        context.begin_state(type("State", (), {"value": "BIN_SIDE_DOCKING"})())
        base = FakeBase()
        depth = FakeDepth()
        navigator = BinSideDockingNavigator(
            load_config(), context, base, depth,
            SequenceDetector([observation(yaw=0.1), observation(yaw=0.0)]),
        )
        settings = self.settings(post_correction_camera_settle_seconds=60.0)
        navigator.step(settings)
        navigator.step(settings)
        waiting = navigator.step(settings)
        self.assertEqual(depth.read_count, 2)
        self.assertEqual(waiting.reason, "exp2 waiting for camera to settle")

    def test_continuous_x_guard_preempts_yaw_and_recenters_before_resuming(self):
        frames = [
            observation(error_x=0.0, yaw=0.3),
            observation(error_x=0.2, yaw=0.3),
            observation(error_x=0.2, yaw=0.3),
            observation(error_x=0.0, yaw=0.3),
            observation(error_x=0.0, yaw=0.3),
        ]
        context, base, navigator = self.runtime_parts(frames)
        settings = self.settings(
            correction_mode="continuous",
            coarse_yaw_center_guard_norm=0.13,
            post_correction_camera_settle_seconds=0.0,
        )
        outcomes = [navigator.step(settings) for _ in frames]
        self.assertEqual(outcomes[1].reason, "exp2 x guard preempted yaw")
        self.assertEqual(
            [motion[2] for motion in base.continuous],
            ["exp2_yaw_align", "exp2_center_align", "exp2_yaw_align"],
        )
        self.assertEqual(context.state_data["exp2_next_axis"], "yaw")

    def test_continuous_yaw_segment_has_configurable_time_limit(self):
        context, base, navigator = self.runtime_parts([observation(error_x=0.0, yaw=0.3)])
        context.state_data["exp2_yaw_correction_started_at"] = time.monotonic() - 2.0
        outcome = navigator.step(self.settings(
            correction_mode="continuous",
            yaw_single_correction_max_seconds=1.0,
        ))
        self.assertEqual(outcome.reason, "exp2 continuous yaw segment time limit")
        self.assertEqual(base.continuous, [])
        self.assertEqual(context.state_data["exp2_next_axis"], "center")

    def test_center_correction_is_followed_by_yaw_recheck(self):
        _, base, navigator = self.runtime_parts([
            observation(error_x=0.2, yaw=0.1),
            observation(error_x=0.2, yaw=0.0),
            observation(error_x=0.2, yaw=0.0),
            observation(error_x=0.0, yaw=0.1),
            observation(error_x=0.0, yaw=0.1),
        ])
        settings = self.settings(stable_frames=5)
        for _ in range(5):
            navigator.step(settings)
        self.assertEqual(
            [motion[2] for motion in base.continuous],
            ["exp2_center_align", "exp2_center_align", "exp2_yaw_align"],
        )
        self.assertEqual(base.pulses, [])


if __name__ == "__main__":
    unittest.main()
