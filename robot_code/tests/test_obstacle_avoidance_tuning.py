import os
import threading
import time
import unittest

from demo_core import TurnResponseModel, load_config
from tuning_tools.obstacle_avoidance_tuning import (
    DEFAULT_PARAMETERS,
    ObstacleAvoidanceSession,
    load_avoidance_parameters,
)


class ObstacleAvoidanceTuningTest(unittest.TestCase):
    def test_standalone_parameters_are_valid(self):
        parameters = load_avoidance_parameters(DEFAULT_PARAMETERS)
        self.assertTrue(os.path.isfile(DEFAULT_PARAMETERS))
        self.assertEqual(parameters["motion"]["side_override"], "auto")
        self.assertGreater(parameters["motion"]["pass_max_seconds"], 0.0)
        self.assertEqual(parameters["control"]["post_turn_clear_frames"], 3)
        self.assertGreater(parameters["control"]["linear_obstacle_confirmation_frames"], 0)
        self.assertNotIn("clear_stable_frames", parameters["control"])

    def test_offset_out_confirms_obstacles_and_stores_partial_segment(self):
        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "motion": {
                "linear_speed": 0.2,
                "linear_meters_per_speed_second": 1.0,
                "minimum_offset_seconds": 0.5,
                "maximum_offset_seconds": 5.0,
            },
            "control": {
                "control_fps": 5.0,
                "blocked_failure_frames": 3,
                "timed_motion_grace_seconds": 1.0,
            },
        }
        session.state = "OFFSET_OUT"
        session.selected_side = "left"
        session.tangent_turn_seconds = 0.7
        session.depth_scaled_segment_seconds = 0.5
        session.offset_segments = []
        session.return_segments = None
        session.estimated_heading_rad = 0.4
        session.estimated_lateral_offset_m = 0.0
        session.lateral_correction_count = 2
        session.log = lambda _message: None
        obstacle = object()
        captured = {}

        def visual(*_args, **kwargs):
            captured.update(kwargs)
            session.motion_interrupted_by_obstacle = obstacle
            return 0.35

        session._run_visual_motion = visual
        session._activate_locked_obstacle = lambda analysis: setattr(session, "state", "DETECTED")

        session._phase_offset_out()

        self.assertTrue(captured["confirm_new_obstacles"])
        self.assertEqual(session.state, "DETECTED")
        self.assertEqual(session.lateral_correction_count, 0)
        self.assertEqual(len(session.offset_segments), 1)
        self.assertAlmostEqual(session.offset_segments[0]["offset_seconds"], 0.35)
        self.assertAlmostEqual(session.offset_segments[0]["turn_seconds"], 0.7)
        self.assertEqual(session.offset_segments[0]["side"], "left")

    def test_linear_motion_stops_and_confirms_new_obstacle(self):
        class Analysis(object):
            status = "CANDIDATE"
            path_blocked = True
            unsafe = False
            detected = True
            reason = "candidate"
            obstacle_depth = 1.8
            bbox = (140, 50, 30, 80)
            mask = type("Mask", (), {"shape": (240, 320)})()
            selected_side = "left"

        class FakeBase(object):
            def __init__(self):
                self.commands = []

            def start_motion(self, direction, speed, label):
                self.commands.append(("start", direction, speed, label))

            def stop(self):
                self.commands.append(("stop",))

        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "control": {
                "missing_frame_limit": 0,
                "linear_obstacle_confirmation_frames": 3,
            },
            "perception": {"tracking_max_center_shift_norm": 0.2},
            "safety": {"emergency_depth": 0.65},
        }
        session.base = FakeBase()
        session.stop_event = threading.Event()
        session.locked_obstacle = None
        session.log = lambda _message: None
        session._sleep_control = lambda: None
        samples = [Analysis(), Analysis(), Analysis()]
        session.sample = lambda: samples.pop(0)

        session._run_visual_motion(
            "forward",
            0.2,
            1.0,
            "pass_obstacle",
            lambda _analysis, _missing, _elapsed: False,
            confirm_new_obstacles=True,
        )

        self.assertEqual(samples, [])
        self.assertIsNotNone(session.motion_interrupted_by_obstacle)
        self.assertEqual(session.locked_obstacle["confirmation_frames"], 3)
        self.assertEqual(
            session.base.commands,
            [("start", "forward", 0.2, "pass_obstacle"), ("stop",), ("stop",)],
        )

    def test_interrupted_offset_return_stores_remaining_distance(self):
        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "motion": {"linear_speed": 0.2},
            "control": {
                "control_fps": 5.0,
                "blocked_failure_frames": 3,
                "timed_motion_grace_seconds": 1.0,
            },
        }
        session.state = "OFFSET_RETURN"
        session.selected_side = "left"
        session.depth_scaled_segment_seconds = 0.0
        session.active_return_segment = {
            "turn_seconds": 0.4,
            "offset_seconds": 1.0,
            "side": "left",
        }
        session.return_segments = []
        session.estimated_lateral_offset_m = 1.0
        session.lateral_correction_count = 2
        session.log = lambda _message: None
        obstacle = object()

        def visual(*_args, **_kwargs):
            session.motion_interrupted_by_obstacle = obstacle
            return 0.4

        session._run_visual_motion = visual
        session._activate_locked_obstacle = lambda analysis: setattr(session, "state", "DETECTED")

        session._phase_offset_return()

        self.assertEqual(session.state, "DETECTED")
        self.assertIsNone(session.active_return_segment)
        self.assertEqual(session.lateral_correction_count, 0)
        self.assertAlmostEqual(session.return_segments[0]["offset_seconds"], 0.6)
        self.assertEqual(session.return_segments[0]["turn_seconds"], 0.0)
        self.assertTrue(session.return_segments[0]["exact_seconds"])

    def test_symmetric_bypass_matches_calibrated_turn_angles_and_offset_durations(self):
        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "motion": {
                "linear_speed": 0.2,
                "turn_speed": 0.15,
                "linear_meters_per_speed_second": 1.0,
                "angular_radians_per_speed_second": 3.14159265359,
                "depth_to_segment_time_factor": 1.0,
                "pass_depth_to_time_factor": 0.75,
                "minimum_offset_seconds": 0.5,
                "maximum_offset_seconds": 2.0,
                "pass_min_seconds": 0.7,
                "pass_max_seconds": 1.5,
                "verification_forward_seconds": 0.3,
            },
            "control": {
                "control_fps": 3.0,
                "blocked_failure_frames": 3,
                "timed_motion_grace_seconds": 1.0,
                "missing_frame_limit": 2,
            },
        }
        session.state = "OFFSET_OUT"
        session.selected_side = "left"
        session.tangent_turn_seconds = 0.42
        session.offset_out_seconds = None
        session.offset_segments = []
        session.return_segments = None
        session.active_return_segment = None
        session.lateral_correction_count = 0
        session.initial_obstacle_depth = 1.2
        session.depth_scaled_segment_seconds = 1.2
        session.pass_depth_scaled_seconds = 0.9
        session.estimated_heading_rad = 0.2
        session.estimated_lateral_offset_m = 0.0
        session.log = lambda _message: None
        session.turn_response = TurnResponseModel(load_config().section("base_turn_response"))
        commands = []

        def timed(direction, speed, seconds, label, require_clear=False):
            commands.append((label, direction, float(speed), float(seconds), bool(require_clear)))
            return float(seconds)

        def visual(direction, speed, timeout, label, stop_condition, minimum_seconds=0.0, **_kwargs):
            commands.append((label, direction, float(speed), float(minimum_seconds), False))
            if label == "offset_out":
                return max(0.73, float(minimum_seconds))
            return float(minimum_seconds)

        session._run_timed_motion = timed
        session._run_visual_motion = visual
        session._verify_clear_while_stopped = lambda _label: True

        session._phase_offset_out()
        session._phase_turn_parallel()
        session._phase_passing()
        session._phase_turn_to_line()
        session._phase_offset_return()
        session._phase_restore_heading()

        turns = [item for item in commands if item[0] in ("turn_parallel", "turn_to_line", "restore_heading")]
        calibrated_right_seconds = 0.42 * 9.0 / 10.4
        self.assertEqual(turns[0][1:3], ("right", 0.15))
        self.assertAlmostEqual(turns[0][3], calibrated_right_seconds)
        self.assertEqual(turns[1][1:3], ("right", 0.15))
        self.assertAlmostEqual(turns[1][3], calibrated_right_seconds)
        self.assertEqual(turns[2][1:4], ("left", 0.15, 0.42))
        offsets = [item for item in commands if item[0] in ("offset_out", "offset_return")]
        self.assertEqual(offsets[0][1:3], offsets[1][1:3])
        self.assertEqual(offsets[0][3], 1.2)
        self.assertEqual(offsets[1][3], 1.2)

        passing = [item for item in commands if item[0] == "pass_obstacle"]
        self.assertEqual(passing[0][3], 0.9)

    def test_passing_uses_independent_depth_duration_and_completes_on_wall(self):
        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "motion": {
                "linear_speed": 0.3,
                "pass_min_seconds": 0.8,
                "pass_max_seconds": 25.0,
            },
            "control": {
                "control_fps": 6.0,
                "blocked_failure_frames": 12,
                "timed_motion_grace_seconds": 1.0,
            },
        }
        session.state = "PASSING"
        session.pass_depth_scaled_seconds = 2.25
        session.log = lambda _message: None
        captured = {}

        def visual(
            direction, speed, timeout, label, stop_condition,
            minimum_seconds=0.0, **kwargs
        ):
            captured.update({
                "direction": direction,
                "speed": speed,
                "timeout": timeout,
                "label": label,
                "minimum": minimum_seconds,
                "stop_condition": stop_condition,
            })
            return minimum_seconds

        session._run_visual_motion = visual
        session._phase_passing()

        self.assertEqual(session.state, "TURN_TO_LINE")
        self.assertEqual(captured["label"], "pass_obstacle")
        self.assertEqual(captured["minimum"], 2.25)
        self.assertTrue(captured["stop_condition"](object(), 0, 2.25))

    def test_depth_scaled_time_uses_locked_obstacle_depth(self):
        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.depth_scaled_segment_seconds = 1.75
        self.assertEqual(session._segment_minimum_seconds(0.5), 1.75)
        self.assertEqual(session._segment_minimum_seconds(2.0), 2.0)

    def test_timed_motion_deadline_allows_post_duration_frame(self):
        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "control": {
                "control_fps": 3.0,
                "blocked_failure_frames": 3,
                "timed_motion_grace_seconds": 1.0,
            }
        }
        session.log = lambda _message: None
        captured = {}

        def visual(direction, speed, timeout, label, stop_condition, minimum_seconds=0.0, **kwargs):
            captured.update({
                "timeout": timeout,
                "minimum": minimum_seconds,
            })
            return minimum_seconds

        session._run_visual_motion = visual
        session._run_timed_motion("left", 0.15, 1.954, "turn_parallel")
        self.assertAlmostEqual(captured["minimum"], 1.954)
        self.assertAlmostEqual(captured["timeout"], 2.954)

    def test_turn_parallel_persistent_block_reenters_tangent_correction(self):
        class BlockedAnalysis(object):
            unsafe = False
            obstacle_depth = 1.8
            path_blocked = True

        class FakeBase(object):
            def stop(self):
                return None

        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "motion": {"turn_speed": 0.15},
            "control": {
                "blocked_failure_frames": 2,
                "maximum_lateral_corrections": 3,
                "missing_frame_limit": 0,
            },
            "safety": {"emergency_depth": 0.65},
        }
        session.state = "TURN_PARALLEL"
        session.selected_side = "left"
        session.tangent_turn_seconds = 0.5
        session.lateral_correction_count = 0
        session.estimated_heading_rad = 0.2
        session.base = FakeBase()
        session.stop_event = threading.Event()
        session.log = lambda _message: None
        session.sample = lambda: BlockedAnalysis()
        session._sleep_control = lambda: None
        session._run_timed_motion = lambda *args, **kwargs: 0.5

        session._phase_turn_parallel()
        self.assertEqual(session.state, "DETECTED")
        self.assertEqual(session.lateral_correction_count, 1)

    def test_single_blocked_shadow_does_not_fail_stopped_verification(self):
        class Analysis(object):
            unsafe = False
            obstacle_depth = 1.8

            def __init__(self, blocked):
                self.path_blocked = blocked

        class FakeBase(object):
            def stop(self):
                return None

        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "control": {
                "blocked_failure_frames": 3,
                "post_turn_clear_frames": 3,
                "missing_frame_limit": 0,
            },
            "safety": {"emergency_depth": 0.65},
        }
        session.base = FakeBase()
        session.stop_event = threading.Event()
        session.log = lambda _message: None
        session._sleep_control = lambda: None
        samples = [Analysis(True), Analysis(False), Analysis(False), Analysis(False)]
        session.sample = lambda: samples.pop(0)

        self.assertTrue(session._verify_clear_while_stopped("test"))

    def test_blocked_streak_can_fail_after_an_earlier_clear_sample(self):
        class Analysis(object):
            unsafe = False
            obstacle_depth = 1.8

            def __init__(self, blocked):
                self.path_blocked = blocked

        class FakeBase(object):
            def stop(self):
                return None

        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "control": {
                "blocked_failure_frames": 3,
                "post_turn_clear_frames": 3,
                "missing_frame_limit": 0,
            },
            "safety": {"emergency_depth": 0.65},
        }
        session.base = FakeBase()
        session.stop_event = threading.Event()
        session.log = lambda _message: None
        session._sleep_control = lambda: None
        samples = [Analysis(False), Analysis(True), Analysis(True), Analysis(True)]
        session.sample = lambda: samples.pop(0)

        self.assertFalse(session._verify_clear_while_stopped("test"))

    def test_post_turn_wall_frames_count_as_clear(self):
        class Analysis(object):
            status = "IGNORED_WALL"
            unsafe = False
            obstacle_depth = 0.4
            path_blocked = False

        class FakeBase(object):
            def stop(self):
                return None

        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "control": {
                "blocked_failure_frames": 12,
                "post_turn_clear_frames": 3,
                "missing_frame_limit": 0,
            },
            "safety": {"emergency_depth": 0.65},
        }
        session.base = FakeBase()
        session.stop_event = threading.Event()
        session.log = lambda _message: None
        session._sleep_control = lambda: None
        samples = [Analysis(), Analysis(), Analysis()]
        session.sample = lambda: samples.pop(0)

        self.assertTrue(session._verify_clear_while_stopped("turn_parallel"))
        self.assertEqual(samples, [])

    def test_tangent_completion_requires_consecutive_safe_frames(self):
        class Analysis(object):
            mask = type("Mask", (), {"shape": (240, 320)})()
            detected = True
            obstacle_depth = 1.8

            def __init__(self, left_norm):
                self.bbox = (int(left_norm * 320), 50, 20, 80)

        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "motion": {
                "turn_speed": 0.15,
                "maximum_tangent_turn_seconds": 6.0,
                "minimum_tangent_turn_seconds": 0.1,
                "angular_radians_per_speed_second": 3.14159265359,
            },
            "perception": {
                "center_corridor_width_norm": 0.24,
                "tangent_margin_norm": 0.04,
                "tracking_max_center_shift_norm": 0.2,
                "tracking_min_area_ratio": 0.45,
                "tracking_max_area_ratio": 2.2,
                "tracking_depth_tolerance_ratio": 0.45,
            },
            "control": {"tangent_safe_frames": 3},
        }
        session.state = "DETECTED"
        session.selected_side = "left"
        session.locked_obstacle = {
            "bbox": (150, 50, 20, 80),
            "tracked_bbox": (150, 50, 20, 80),
            "obstacle_depth": 1.8,
        }
        session.log = lambda _message: None
        calls = []

        def visual(
            _direction, _speed, _timeout, _label, stop_condition,
            minimum_seconds=0.0, ignore_obstacle_failures=False,
        ):
            self.assertTrue(ignore_obstacle_failures)
            for left_norm in (0.67, 0.50, 0.67, 0.68, 0.69):
                calls.append(left_norm)
                if stop_condition(Analysis(left_norm), 0, 1.0):
                    return 1.0
            raise AssertionError("tangent condition did not complete")

        session._run_visual_motion = visual
        session._phase_turn_to_tangent()
        self.assertEqual(calls, [0.67, 0.50, 0.67, 0.68, 0.69])
        self.assertEqual(session.state, "OFFSET_OUT")

    def test_multiple_outbound_segments_are_replayed_in_reverse(self):
        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "motion": {
                "linear_speed": 0.2,
                "turn_speed": 0.15,
                "angular_radians_per_speed_second": 3.14159265359,
            },
            "control": {
                "control_fps": 6.0,
                "blocked_failure_frames": 3,
                "timed_motion_grace_seconds": 1.0,
            },
        }
        session.state = "TURN_TO_LINE"
        session.selected_side = "left"
        session.depth_scaled_segment_seconds = 0.0
        session.offset_segments = [
            {"turn_seconds": 0.4, "offset_seconds": 0.8, "side": "left"},
            {"turn_seconds": 0.6, "offset_seconds": 1.2, "side": "right"},
        ]
        session.return_segments = None
        session.active_return_segment = None
        session.estimated_heading_rad = 0.0
        session.estimated_lateral_offset_m = 1.0
        session.log = lambda _message: None
        commands = []

        def timed(direction, speed, seconds, label, require_clear=False):
            commands.append((label, direction, float(seconds)))
            return float(seconds)

        session._run_timed_motion = timed
        session._run_visual_motion = lambda direction, speed, timeout, label, stop_condition, minimum_seconds=0.0, **kwargs: timed(
            direction, speed, minimum_seconds, label
        )
        for _index in range(2):
            session._phase_turn_to_line()
            session._phase_offset_return()
            session._phase_restore_heading()

        self.assertEqual(commands, [
            ("turn_to_line", "left", 0.6),
            ("offset_return", "forward", 1.2),
            ("restore_heading", "right", 0.6),
            ("turn_to_line", "right", 0.4),
            ("offset_return", "forward", 0.8),
            ("restore_heading", "left", 0.4),
        ])
        self.assertEqual(session.state, "VERIFY_CLEAR")

    def test_candidate_stops_then_resumes_when_bbox_leaves_center(self):
        class Analysis(object):
            def __init__(self, status, path_blocked, unsafe=False, reason=None):
                self.status = status
                self.path_blocked = path_blocked
                self.unsafe = unsafe
                self.reason = reason or status.lower()
                self.obstacle_depth = 1.8
                self.bbox = (140, 50, 30, 80)
                self.mask = type("Mask", (), {"shape": (240, 320)})()
                self.selected_side = "left"

        class FakeBase(object):
            def __init__(self):
                self.commands = []

            def stop(self):
                self.commands.append(("stop",))

            def start_motion(self, direction, speed, label):
                self.commands.append((direction, speed, label))

        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "control": {"obstacle_confirmation_frames": 3},
            "perception": {},
            "safety": {"emergency_depth": 0.65},
        }
        session.base = FakeBase()
        session.stop_event = threading.Event()
        session.locked_obstacle = None
        session.log = lambda _message: None
        session._sleep_control = lambda: None
        samples = [Analysis("OBSTACLE", False)]
        session.sample = lambda: samples.pop(0)

        resolved = session._confirm_candidate_while_stopped(
            Analysis("CANDIDATE", True), "forward", 0.2, "avoidance_cruise"
        )
        self.assertEqual(resolved.status, "OBSTACLE")
        self.assertEqual(
            session.base.commands,
            [("stop",), ("forward", 0.2, "avoidance_cruise_resume")],
        )

    def test_candidate_is_confirmed_after_consecutive_center_frames(self):
        class Analysis(object):
            def __init__(self):
                self.status = "CANDIDATE"
                self.path_blocked = True
                self.unsafe = False
                self.reason = "candidate"
                self.obstacle_depth = 1.8
                self.bbox = (140, 50, 30, 80)
                self.mask = type("Mask", (), {"shape": (240, 320)})()
                self.selected_side = "left"

        class FakeBase(object):
            def __init__(self):
                self.commands = []

            def stop(self):
                self.commands.append(("stop",))

            def start_motion(self, direction, speed, label):
                self.commands.append((direction, speed, label))

        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "control": {"obstacle_confirmation_frames": 3},
            "perception": {},
            "safety": {"emergency_depth": 0.65},
        }
        session.base = FakeBase()
        session.stop_event = threading.Event()
        session.locked_obstacle = None
        session.log = lambda _message: None
        session._sleep_control = lambda: None
        samples = [Analysis(), Analysis()]
        session.sample = lambda: samples.pop(0)

        confirmed = session._confirm_candidate_while_stopped(
            Analysis(), "forward", 0.2, "avoidance_cruise"
        )
        self.assertEqual(confirmed.status, "OBSTACLE")
        self.assertTrue(confirmed.path_blocked)
        self.assertEqual(session.locked_obstacle["confirmation_frames"], 3)
        self.assertEqual(session.locked_obstacle["bbox"], (140, 50, 30, 80))
        self.assertEqual(session.base.commands, [("stop",)])

    def test_candidate_identity_change_cancels_lock_and_resumes(self):
        class Analysis(object):
            mask = type("Mask", (), {"shape": (240, 320)})()
            status = "CANDIDATE"
            path_blocked = True
            unsafe = False
            reason = "candidate"
            obstacle_depth = 1.8
            selected_side = "left"

            def __init__(self, bbox):
                self.bbox = bbox

        class FakeBase(object):
            def __init__(self):
                self.commands = []

            def stop(self):
                self.commands.append(("stop",))

            def start_motion(self, direction, speed, label):
                self.commands.append((direction, speed, label))

        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "control": {"obstacle_confirmation_frames": 3},
            "perception": {"tracking_max_center_shift_norm": 0.2},
            "safety": {"emergency_depth": 0.65},
        }
        session.base = FakeBase()
        session.stop_event = threading.Event()
        session.locked_obstacle = None
        session.log = lambda _message: None
        session._sleep_control = lambda: None
        session.sample = lambda: Analysis((270, 50, 30, 80))

        resolved = session._confirm_candidate_while_stopped(
            Analysis((140, 50, 30, 80)), "forward", 0.2, "avoidance_cruise"
        )
        self.assertIsNone(session.locked_obstacle)
        self.assertEqual(resolved.bbox, (270, 50, 30, 80))
        self.assertFalse(resolved.detected)
        self.assertFalse(resolved.path_blocked)
        self.assertEqual(resolved.status, "UNCONFIRMED")
        self.assertEqual(
            session.base.commands,
            [("stop",), ("forward", 0.2, "avoidance_cruise_resume")],
        )

    def test_missing_locked_bbox_counts_as_safe_and_center_bbox_resets_streak(self):
        class Analysis(object):
            mask = type("Mask", (), {"shape": (240, 320)})()
            obstacle_depth = 1.8

            def __init__(self, bbox=None, detected=True, candidate_bboxes=None, status="CLEAR"):
                self.bbox = bbox
                self.detected = detected
                self.candidate_bboxes = list(candidate_bboxes or [])
                self.status = status

        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "motion": {
                "turn_speed": 0.15,
                "maximum_tangent_turn_seconds": 6.0,
                "minimum_tangent_turn_seconds": 0.1,
                "angular_radians_per_speed_second": 3.14159265359,
            },
            "perception": {
                "center_corridor_width_norm": 0.24,
                "tangent_margin_norm": 0.04,
                "tracking_max_center_shift_norm": 0.2,
                "tracking_min_area_ratio": 0.45,
                "tracking_max_area_ratio": 2.2,
                "tracking_depth_tolerance_ratio": 0.45,
            },
            "control": {"tangent_safe_frames": 3},
        }
        session.state = "DETECTED"
        session.selected_side = "left"
        session.locked_obstacle = {
            "bbox": (150, 50, 20, 80),
            "tracked_bbox": (150, 50, 20, 80),
            "obstacle_depth": 1.8,
        }
        session.log = lambda _message: None
        results = []

        def visual(
            _direction, _speed, _timeout, _label, stop_condition,
            minimum_seconds=0.0, ignore_obstacle_failures=False,
        ):
            self.assertTrue(ignore_obstacle_failures)
            samples = (
                Analysis(None, detected=False),
                Analysis((0, 20, 320, 100), detected=False, status="IGNORED_WALL"),
                Analysis((160, 50, 20, 80)),
                Analysis((280, 20, 30, 60)),
                Analysis((0, 20, 320, 100), detected=False, status="IGNORED_WALL"),
                Analysis(
                    (280, 20, 30, 60),
                    candidate_bboxes=[(280, 20, 30, 60), (214, 50, 20, 80)],
                ),
                Analysis(None, detected=False),
            )
            for sample in samples:
                results.append(stop_condition(sample, 1 if sample.bbox is None else 0, 1.0))
                if results[-1]:
                    return 1.0
            raise AssertionError("tracked tangent did not complete")

        session._run_visual_motion = visual
        session._phase_turn_to_tangent()
        self.assertEqual(results, [False, False, False, False, False, True])

    def test_rotation_motion_does_not_fail_for_unrelated_blocked_geometry(self):
        class Analysis(object):
            status = "BLOCKED"
            unsafe = True
            detected = True
            path_blocked = True
            obstacle_depth = 0.5
            reason = "new obstacle"

        class FakeBase(object):
            def __init__(self):
                self.commands = []

            def start_motion(self, direction, speed, label):
                self.commands.append((direction, speed, label))

            def stop(self):
                self.commands.append(("stop",))

        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "control": {"missing_frame_limit": 0},
            "safety": {"emergency_depth": 0.65},
        }
        session.base = FakeBase()
        session.stop_event = threading.Event()
        session.sample = lambda: Analysis()
        session._sleep_control = lambda: None

        elapsed = session._run_visual_motion(
            "left", 0.15, 1.0, "turn_to_tangent",
            lambda _analysis, _missing, _elapsed: True,
            ignore_obstacle_failures=True,
        )
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(
            session.base.commands,
            [("left", 0.15, "turn_to_tangent"), ("stop",)],
        )

    def test_rotation_still_stops_for_wall_absolute_depth_protection(self):
        class Analysis(object):
            status = "BLOCKED"
            unsafe = True
            detected = True
            path_blocked = True
            obstacle_depth = 1.0
            reason = "wide wall reached absolute center stop"

        class FakeBase(object):
            def __init__(self):
                self.commands = []

            def start_motion(self, direction, speed, label):
                self.commands.append((direction, speed, label))

            def stop(self):
                self.commands.append(("stop",))

        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "control": {"missing_frame_limit": 0},
            "safety": {"emergency_depth": 0.65},
        }
        session.base = FakeBase()
        session.stop_event = threading.Event()
        session.sample = lambda: Analysis()
        session._sleep_control = lambda: None

        with self.assertRaisesRegex(RuntimeError, "absolute depth protection"):
            session._run_visual_motion(
                "left", 0.15, 1.0, "turn_to_tangent",
                lambda _analysis, _missing, _elapsed: False,
                ignore_obstacle_failures=True,
            )
        self.assertEqual(
            session.base.commands,
            [("left", 0.15, "turn_to_tangent"), ("stop",)],
        )

    def test_wall_frame_reaches_generic_motion_stop_condition_as_clear(self):
        class Analysis(object):
            unsafe = False
            path_blocked = False
            obstacle_depth = 0.4
            reason = "test"

            def __init__(self, status, detected):
                self.status = status
                self.detected = detected

        class FakeBase(object):
            def start_motion(self, _direction, _speed, _label):
                return None

            def stop(self):
                return None

        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "control": {"missing_frame_limit": 0},
            "safety": {"emergency_depth": 0.65},
        }
        session.base = FakeBase()
        session.stop_event = threading.Event()
        session.log = lambda _message: None
        session._sleep_control = lambda: None
        samples = [Analysis("IGNORED_WALL", False)]
        session.sample = lambda: samples.pop(0)
        decisions = []

        session._run_visual_motion(
            "forward", 0.2, 1.0, "test_motion",
            lambda analysis, _missing, _elapsed: decisions.append(analysis.status) or True,
        )
        self.assertEqual(decisions, ["IGNORED_WALL"])

    def test_wall_frame_does_not_extend_pure_timed_motion(self):
        class Analysis(object):
            status = "IGNORED_WALL"
            unsafe = False
            detected = False
            path_blocked = False
            obstacle_depth = 1.8
            reason = "center bbox exceeds wall width threshold"

        class FakeBase(object):
            def __init__(self):
                self.commands = []

            def start_motion(self, direction, speed, label):
                self.commands.append((direction, speed, label))

            def stop(self):
                self.commands.append(("stop",))

        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.parameters = {
            "control": {
                "control_fps": 6.0,
                "blocked_failure_frames": 12,
                "timed_motion_grace_seconds": 1.0,
                "missing_frame_limit": 0,
            },
            "safety": {"emergency_depth": 0.65},
        }
        session.base = FakeBase()
        session.stop_event = threading.Event()
        session.log = lambda _message: None
        session.sample = lambda: Analysis()
        session._sleep_control = lambda: None

        elapsed = session._run_timed_motion("right", 0.15, 0.0, "turn_parallel")
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(
            session.base.commands,
            [("right", 0.15, "turn_parallel"), ("stop",)],
        )

    def test_motion_job_replaces_stationary_live_monitor(self):
        class FakeBase(object):
            def stop(self):
                return None

        session = ObstacleAvoidanceSession.__new__(ObstacleAvoidanceSession)
        session.worker_lock = threading.Lock()
        session.stop_event = threading.Event()
        session.worker = None
        session.worker_name = None
        session.base = FakeBase()
        session.state = "READY"
        session.log = lambda _message: None
        session._close_writer = lambda: None
        events = []

        def monitor():
            events.append("monitor_started")
            while not session.stop_event.wait(0.01):
                pass
            events.append("monitor_stopped")

        session.start_job(monitor, "live_monitor")
        deadline = time.time() + 1.0
        while "monitor_started" not in events and time.time() < deadline:
            time.sleep(0.01)
        session.start_job(lambda: events.append("motion_started"), "full_bypass", replace_live=True)
        worker = session.worker
        if worker is not None:
            worker.join(1.0)
        self.assertEqual(events, ["monitor_started", "monitor_stopped", "motion_started"])


if __name__ == "__main__":
    unittest.main()
