import os
import threading
import time
import unittest

import numpy as np

from demo_core import DemoStateMachine, MissionEvent, MissionState, RobotComponents, TargetType, load_config


class FakeBase(object):
    def __init__(self, events=None):
        self.commands = []
        self.motion_tracker = None
        self.active_motion = None
        self.events = events

    def attach_motion_tracker(self, tracker):
        self.motion_tracker = tracker

    def pulse(self, direction, speed, seconds, label):
        self.commands.append((label, direction, float(speed), float(seconds)))
        if self.events is not None:
            self.events.append(("base", label))
        if self.motion_tracker is not None:
            self.motion_tracker.record_motion(direction, speed, seconds)

    def start_motion(self, direction, speed, label):
        self.active_motion = (label, direction, float(speed))
        self.commands.append((label, direction, float(speed), "continuous"))

    def motion_active(self, label=None):
        return self.active_motion is not None and (label is None or self.active_motion[0] == label)

    def update_motion_odometry(self, dry_run_step_seconds=0.0):
        return 0.0

    def stop(self):
        self.active_motion = None
        self.commands.append(("stop",))


class FakeArm(object):
    def __init__(self, events=None):
        self.poses = []
        self.events = events
        self.cancel_count = 0
        self.cancelled = False

    def pose(self, name, pose=None):
        self.cancelled = False
        self.poses.append((name, dict(pose or {})))
        if self.events is not None:
            self.events.append(("arm_pose", name))
        return {1: 500}

    def wait_for_positions(self, targets, label):
        if self.events is not None:
            self.events.append(("arm_settled", label))
        return True

    def cancel_motion(self):
        self.cancel_count += 1
        self.cancelled = True

    def motion_cancelled(self):
        return self.cancelled


class FakeDepth(object):
    def __init__(self):
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.started = False
        self.depth_map_calls = 0
        self.stop_count = 0

    def start(self):
        self.started = True

    def stop(self):
        self.started = False
        self.stop_count += 1

    def read_frame(self):
        return self.frame

    def obstacle_detected_frame(self, frame):
        return False

    def observe_lens_center_frame(self, frame):
        return {"mean": 1.0, "min": 1.0, "max": 1.0}

    def depth_map_frame(self, frame):
        self.depth_map_calls += 1
        return np.full((240, 320), 2.0, dtype=np.float32)

    def observe_center_depth_map(self, label, depth_map, center_x, center_y, image_width, image_height, width_ratio, height_ratio, source_space="raw"):
        return {"mean": 0.5, "min": 0.5, "max": 0.5}

    def grab_verified(self):
        return True


class ObstacleDepth(FakeDepth):
    def __init__(self):
        super(ObstacleDepth, self).__init__()
        self.blocked = True

    def obstacle_detected_frame(self, frame):
        return self.blocked


class FakeDetector(object):
    def __init__(self, kind):
        self.kind = kind

    def detect(self, frame):
        return {
            "kind": self.kind,
            "found": True,
            "confidence": 0.9,
            "bbox": [120.0, 40.0, 200.0, 180.0],
            "bbox_height_norm": 0.58,
            "center_x": 160.0,
            "center_y": 110.0,
            "error_x": 0.0,
            "vertical_edge_error": 0.0,
            "horizontal_edge_error": 0.0,
            "max_corner_angle_error_deg": 0.0,
            "yaw_error_rad": 0.0,
            "yaw_error_deg": 0.0,
            "yaw_source": "test",
            "pose": {"x": 0.0, "y": 0.0, "z": 0.19},
            "distance": 0.19,
        }

    def detect_all(self, frame):
        return [self.detect(frame)]

    def confidence_threshold(self, tracking=False):
        return 0.2


class MissingDetector(FakeDetector):
    def detect(self, frame):
        return {"kind": self.kind, "found": False, "confidence": 0.0}

    def detect_all(self, frame):
        return []


class SlowMissingDetector(MissingDetector):
    def detect(self, frame):
        time.sleep(0.01)
        return super(SlowMissingDetector, self).detect(frame)


class StateMachineTest(unittest.TestCase):
    def test_rotate_search_ignores_obstacles(self):
        config = load_config(overrides={
            "avoidance": {"strategy": "scripted"},
            "navigation": {"can": {"search": {"routine": 0}}},
        })
        runtime = DemoStateMachine(
            config,
            services=RobotComponents(
                FakeBase(), FakeArm(), ObstacleDepth(), MissingDetector("can"), FakeDetector("bin_tag")
            ),
        )
        runtime.state = MissionState.SEARCHING
        runtime.context.target_type = TargetType.CAN
        runtime.context.begin_state(runtime.state)

        outcome = runtime.step_once()

        self.assertIsNone(outcome.event)
        self.assertEqual(runtime.state, MissionState.SEARCHING)
        self.assertEqual(runtime.services.base.active_motion[1], "left")

    def test_square_search_avoidance_resumes_same_routine_with_displacement_memory(self):
        config = load_config(overrides={
            "avoidance": {"strategy": "scripted"},
            "navigation": {"can": {"search": {"routine": 2}}},
        })
        depth = ObstacleDepth()
        runtime = DemoStateMachine(
            config,
            services=RobotComponents(
                FakeBase(), FakeArm(), depth, MissingDetector("can"), FakeDetector("bin_tag")
            ),
        )
        runtime.state = MissionState.SEARCHING
        runtime.context.target_type = TargetType.CAN
        runtime.context.begin_state(runtime.state)

        blocked = runtime.step_once()
        self.assertEqual(blocked.event, MissionEvent.OBSTACLE_FOUND)
        self.assertEqual(runtime.state, MissionState.AVOIDING)
        memory = runtime.context.searching_routine_data["searching_routine"]
        self.assertTrue(memory["paused_for_avoidance"])

        depth.blocked = False
        resumed = runtime.step_once()
        self.assertEqual(resumed.event, MissionEvent.ROUTINE_RESUME)
        self.assertEqual(runtime.state, MissionState.SEARCHING)
        memory = runtime.context.searching_routine_data["searching_routine"]
        self.assertFalse(memory["paused_for_avoidance"])
        self.assertEqual(len(memory["avoidance_history"]), 1)
        self.assertNotEqual(memory["avoidance_history"][0]["dx_m"], 0.0)

    def test_transition_exposes_read_only_hud_metadata(self):
        config = load_config(overrides={"vague_map": {"enabled": False}})
        runtime = DemoStateMachine(
            config,
            services=RobotComponents(FakeBase(), FakeArm(), FakeDepth(), FakeDetector("can"), FakeDetector("bin_tag")),
        )
        runtime.transition(MissionEvent.START, "operator start")
        self.assertEqual(runtime.last_transition["from_state"], "IDLE")
        self.assertEqual(runtime.last_transition["event"], "START")
        self.assertEqual(runtime.last_transition["to_state"], "INITIALIZING")
        self.assertEqual(runtime.last_transition["reason"], "operator start")

    def test_stop_request_is_graceful_and_worker_can_own_cleanup(self):
        config = load_config(overrides={
            "runtime": {"loop_pause_seconds": 0.0, "max_pickups": 1},
            "vague_map": {"enabled": False},
            "arm": {"poses": {"safe_home": {"pause_seconds": 0.0}}},
        })
        base = FakeBase()
        arm = FakeArm()
        depth = FakeDepth()
        runtime = DemoStateMachine(
            config,
            services=RobotComponents(
                base,
                arm,
                depth,
                SlowMissingDetector("can"),
                FakeDetector("bin_tag"),
            ),
        )
        result = []
        worker = threading.Thread(
            target=lambda: result.append(runtime.run(max_ticks=10000, cleanup=False))
        )
        worker.start()
        deadline = time.time() + 1.0
        while runtime.state != MissionState.SEARCHING and time.time() < deadline:
            time.sleep(0.005)

        runtime.request_stop()
        worker.join(2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [False])
        self.assertNotEqual(runtime.state, MissionState.FAILED)
        self.assertIsNone(runtime.context.last_error)
        self.assertEqual(arm.cancel_count, 1)
        self.assertEqual(depth.stop_count, 0)
        runtime.stop_all()
        self.assertEqual(depth.stop_count, 1)
        cleanup_pose_count = len([name for name, _ in arm.poses if name == "safe_home"])
        runtime.stop_all()
        self.assertEqual(depth.stop_count, 1)
        self.assertEqual(len([name for name, _ in arm.poses if name == "safe_home"]), cleanup_pose_count)

    def test_delivery_asset_paths_resolve_from_config_directory(self):
        config = load_config()

        settings = config.section("detectors")["can"]
        self.assertTrue(config.resolve_path(settings["model_path"]).endswith(
            os.path.join("assets", "models", "detectnet_native_can", "can_ssd_mobilenet_v1.onnx")
        ))
        self.assertTrue(os.path.exists(config.resolve_path(settings["model_path"])))
        self.assertTrue(os.path.exists(config.resolve_path(settings["labels_path"])))
        self.assertTrue(config.get("runtime.dry_run.arm"))

    def test_command_overrides_take_precedence_over_runtime_config(self):
        config = load_config(overrides={
            "runtime": {"dry_run": {"arm": False}},
        })
        self.assertFalse(config.get("runtime.dry_run.arm"))

    def test_full_can_to_bin_mission_reaches_done_and_counts_pickup(self):
        pose_overrides = {}
        for name in ("safe_home", "arm_down", "grab", "carry", "release"):
            pose_overrides[name] = {"pause_seconds": 0.0}
        config = load_config(overrides={
            "runtime": {"loop_pause_seconds": 0.0, "max_pickups": 1},
            "vague_map": {"enabled": False},
            "navigation": {
                "can": {"near_align": {"required_stable_frames": 2}},
                "bin": {
                    "align": {"target_error_x_norm": 0.0},
                    "near_align": {"required_stable_frames": 2},
                    "side_docking": {"experimental": {"enabled": False}},
                },
            },
            "arm": {
                "pickup_start_delay_seconds": 0.0,
                "push": {"speed": "linear.slow", "seconds": 0.0, "post_lock_seconds": 0.0},
                "poses": pose_overrides,
            },
        })
        base = FakeBase()
        arm = FakeArm()
        depth = FakeDepth()
        services = RobotComponents(base, arm, depth, FakeDetector("can"), FakeDetector("bin_tag"))
        runtime = DemoStateMachine(config, services=services)

        self.assertTrue(runtime.run(max_ticks=100))
        self.assertEqual(runtime.state, MissionState.DONE)
        self.assertEqual(runtime.context.completed_pickups, 1)
        self.assertFalse(runtime.context.grabbed)
        self.assertGreater(runtime.context.metrics["detector_calls"]["can"], 0)
        self.assertGreater(runtime.context.metrics["detector_calls"]["bin"], 0)
        self.assertIn("arm_down", [name for name, _ in arm.poses])
        self.assertIn("release", [name for name, _ in arm.poses])

    def test_exp2_side_docking_waits_for_arm_then_side_parks(self):
        pose_overrides = {}
        for name in (
            "safe_home", "arm_down", "grab", "carry", "release",
            "side_view_grabbing", "side_view_bin_insert",
        ):
            pose_overrides[name] = {"pause_seconds": 0.0}
        config = load_config(overrides={
            "runtime": {"loop_pause_seconds": 0.0, "max_pickups": 1},
            "vague_map": {"enabled": False},
            "navigation": {
                "can": {"near_align": {"required_stable_frames": 1}},
                "bin": {
                    "align": {"target_error_x_norm": 0.0},
                    "near_align": {"required_stable_frames": 1},
                    "side_docking": {"experimental": {
                        "enabled": True,
                        "stable_frames": 2,
                        "post_entry_camera_settle_seconds": 0.0,
                        "post_entry_camera_discard_frames": 0,
                        "post_correction_camera_settle_seconds": 0.0,
                    }},
                },
            },
            "arm": {
                "pickup_start_delay_seconds": 0.0,
                "push": {"speed": "linear.slow", "seconds": 0.0, "post_lock_seconds": 0.0},
                "poses": pose_overrides,
            },
        })
        events = []
        base = FakeBase(events)
        arm = FakeArm(events)
        services = RobotComponents(base, arm, FakeDepth(), FakeDetector("can"), FakeDetector("bin_tag"))
        runtime = DemoStateMachine(config, services=services)

        self.assertTrue(runtime.run(max_ticks=120))
        pose_names = [name for name, _ in arm.poses]
        self.assertIn("side_view_grabbing", pose_names)
        self.assertIn("side_view_bin_insert", pose_names)
        self.assertIn("release", pose_names)
        self.assertNotIn("exp2_supportive_base_turn", [command[0] for command in base.commands])
        self.assertIn(("exp2_side_entry_turn", "left", 0.15, 10.4), base.commands)
        self.assertNotIn("exp2_reset_heading", [command[0] for command in base.commands])
        self.assertLess(events.index(("arm_settled", "side_view_grabbing")), events.index(("base", "exp2_side_entry_turn")))
        release_settled_index = events.index(("arm_settled", "release"))
        insert_settled_index = events.index(("arm_settled", "side_view_bin_insert"))
        final_safe_home_index = events.index(("arm_pose", "safe_home"), release_settled_index)
        self.assertLess(insert_settled_index, release_settled_index)
        self.assertLess(release_settled_index, final_safe_home_index)
        side_view_index = events.index(("arm_pose", "side_view_grabbing"))
        self.assertNotIn(("arm_pose", "carry"), events[side_view_index:release_settled_index])

    def test_exp2_side_docking_failure_restores_carry_view_and_heading(self):
        config = load_config(overrides={
            "runtime": {"loop_pause_seconds": 0.0, "max_pickups": 1},
            "vague_map": {"enabled": False},
            "navigation": {"bin": {"side_docking": {"experimental": {
                "enabled": True,
                "lost_frame_limit": 1,
                "post_entry_camera_settle_seconds": 0.0,
                "post_entry_camera_discard_frames": 0,
            }}}},
            "arm": {"poses": {
                "side_view_grabbing": {"pause_seconds": 0.0},
                "carry": {"pause_seconds": 0.0},
            }},
        })
        base = FakeBase()
        arm = FakeArm()
        runtime = DemoStateMachine(
            config,
            services=RobotComponents(base, arm, FakeDepth(), FakeDetector("can"), MissingDetector("bin_tag")),
        )
        runtime.context.target_type = TargetType.BIN
        runtime.context.grabbed = True
        runtime.state = MissionState.BIN_SIDE_DOCKING
        runtime.context.begin_state(runtime.state)
        runtime.context.last_observation = FakeDetector("bin_tag").detect(None)

        runtime.step_once()
        outcome = runtime.step_once()
        self.assertEqual(outcome.event, MissionEvent.TARGET_MISSING)
        self.assertIn("carry", [name for name, _ in arm.poses])
        self.assertNotIn("exp2_abort_reset_heading", [command[0] for command in base.commands])

    def test_exp2_side_entry_turns_opposite_arm_by_90_and_keeps_servo1_fixed(self):
        base = FakeBase()
        arm = FakeArm()
        runtime = DemoStateMachine(
            load_config(overrides={"vague_map": {"enabled": False}}),
            services=RobotComponents(
                base, arm, FakeDepth(), FakeDetector("can"), FakeDetector("bin_tag")
            ),
        )
        runtime.context.target_type = TargetType.BIN
        runtime.context.grabbed = True
        runtime.state = MissionState.BIN_SIDE_DOCKING
        runtime.context.begin_state(runtime.state)
        runtime.context.last_observation = {
            "found": True,
            "yaw_error_rad": 0.2,
            "yaw_source": "test",
        }

        outcome = runtime.step_once()

        expected_angle = 0.5 * np.pi
        expected_seconds = 10.4
        expected_servo = 70
        self.assertIsNone(outcome.event)
        entry_command = [command for command in base.commands if command[0] == "exp2_side_entry_turn"][0]
        self.assertEqual(entry_command[1:3], ("left", 0.15))
        self.assertAlmostEqual(entry_command[3], expected_seconds)
        side_pose = [pose for name, pose in arm.poses if name == "side_view_grabbing"][0]
        self.assertEqual(side_pose["s1"], expected_servo)

    def test_exp2_ignores_camera_until_entry_settles_and_discards_frames(self):
        class CountingDetector(FakeDetector):
            def __init__(self, kind):
                super(CountingDetector, self).__init__(kind)
                self.calls = 0

            def detect(self, frame):
                self.calls += 1
                return super(CountingDetector, self).detect(frame)

        detector = CountingDetector("bin_tag")
        runtime = DemoStateMachine(
            load_config(overrides={
                "runtime": {"loop_pause_seconds": 0.0},
                "camera": {"observation_pause_seconds": 0.0},
                "vague_map": {"enabled": False},
                "navigation": {"bin": {"side_docking": {"experimental": {
                    "post_entry_camera_settle_seconds": 60.0,
                    "post_entry_camera_discard_frames": 2,
                }}}},
                "arm": {"poses": {"side_view_grabbing": {"pause_seconds": 0.0}}},
            }),
            services=RobotComponents(
                FakeBase(), FakeArm(), FakeDepth(), FakeDetector("can"), detector
            ),
        )
        runtime.context.target_type = TargetType.BIN
        runtime.context.grabbed = True
        runtime.state = MissionState.BIN_SIDE_DOCKING
        runtime.context.begin_state(runtime.state)

        runtime.step_once()
        runtime.step_once()
        self.assertEqual(detector.calls, 0)

        runtime.context.state_data["entry_camera_ready_at"] = 0.0
        runtime.context.state_data["entry_camera_next_discard_at"] = 0.0
        runtime.step_once()
        runtime.step_once()
        self.assertEqual(detector.calls, 0)

        runtime.step_once()
        self.assertEqual(detector.calls, 1)

    def test_bin_map_arrival_transitions_into_400_degree_scan(self):
        base = FakeBase()
        runtime = DemoStateMachine(
            load_config(),
            services=RobotComponents(
                base, FakeArm(), FakeDepth(), FakeDetector("can"), MissingDetector("bin_tag")
            ),
        )
        runtime.context.target_type = TargetType.BIN
        runtime.context.grabbed = True
        runtime.state = MissionState.MAP_NAVIGATING
        runtime.context.begin_state(runtime.state)
        runtime.map_navigator.step_toward = lambda destination, label: True

        arrival = runtime.step_once()
        scan = runtime.step_once()

        self.assertEqual(arrival.event, MissionEvent.MAP_DESTINATION_REACHED)
        self.assertEqual(runtime.state, MissionState.SEARCHING)
        self.assertIsNone(scan.event)
        self.assertEqual(runtime.current_action, "map_bin_arrival_scan")
        self.assertEqual(
            base.active_motion,
            ("search_bin_routine_4_step_0", "left", 0.3),
        )

    def test_intermediate_retries_then_exhausts_limit(self):
        config = load_config(overrides={"runtime": {"retry_limit": 1}})
        runtime = DemoStateMachine(
            config,
            services=RobotComponents(
                FakeBase(), FakeArm(), FakeDepth(), FakeDetector("can"), FakeDetector("bin_tag")
            ),
        )
        runtime.previous_state = MissionState.SEARCHING
        first = runtime._handle_intermediate_state()
        second = runtime._handle_intermediate_state()
        self.assertEqual(first.event, MissionEvent.RETRY)
        self.assertEqual(second.event, MissionEvent.RETRY_EXHAUSTED)

    def test_map_search_timeout_replans_then_restarts_searching(self):
        config = load_config()
        runtime = DemoStateMachine(
            config,
            services=RobotComponents(
                FakeBase(), FakeArm(), FakeDepth(), MissingDetector("can"), FakeDetector("bin_tag")
            ),
        )
        runtime.state = MissionState.SEARCHING
        runtime.context.target_type = TargetType.CAN
        runtime.context.begin_state(MissionState.SEARCHING)
        runtime.context.state_data["entered_at"] = time.time() - 4000.0
        outcome = runtime.step_once()
        self.assertEqual(outcome.event, MissionEvent.REPLAN)
        self.assertEqual(runtime.state, MissionState.PLANNING)
        runtime.step_once()
        self.assertEqual(runtime.state, MissionState.SEARCHING)

    def test_map_navigation_visual_detection_takes_control_before_motion(self):
        from demo_core.vague_map import Point2D

        config = load_config()
        base = FakeBase()
        services = RobotComponents(base, FakeArm(), FakeDepth(), FakeDetector("can"), FakeDetector("bin_tag"))
        runtime = DemoStateMachine(config, services=services)
        mapped = runtime.vague_map.remember_can(Point2D(0.6, 0.0), 0.8)
        runtime.vague_map.selected_can_id = mapped.can_id
        runtime.context.target_type = TargetType.CAN
        runtime.state = MissionState.MAP_NAVIGATING
        runtime.context.begin_state(runtime.state)
        outcome = runtime.step_once()
        self.assertEqual(outcome.event, MissionEvent.TARGET_FOUND)
        self.assertEqual(runtime.state, MissionState.ALIGNING)
        self.assertFalse(any(command[0].startswith("map_") for command in base.commands if len(command) > 1))

    def test_multiple_incidental_cans_use_one_depth_inference_without_height_filter(self):
        config = load_config(overrides={
            "vague_map": {
                "depth_to_distance_scale_m_per_unit": 0.25,
                "can_merge_radius_m": 0.01,
            },
        })
        depth = FakeDepth()
        runtime = DemoStateMachine(
            config,
            services=RobotComponents(FakeBase(), FakeArm(), depth, FakeDetector("can"), FakeDetector("bin_tag")),
        )
        frame = depth.read_frame()
        lower = FakeDetector("can").detect(frame)
        lower["center_y"] = 180.0
        upper = dict(lower)
        upper["center_x"] = 220.0
        upper["center_y"] = 40.0
        upper["error_x"] = 0.5
        runtime._map_can_detections(frame, [lower, upper])
        self.assertEqual(depth.depth_map_calls, 1)
        self.assertEqual(len(runtime.vague_map.known_cans), 2)

    def test_incidental_can_confirmation_uses_half_normal_final_verify_frames(self):
        config = load_config(overrides={
            "navigation": {"can": {"near_align": {"required_stable_frames": 4}}},
            "vague_map": {"depth_to_distance_scale_m_per_unit": 0.25},
        })
        depth = FakeDepth()
        detector = FakeDetector("can")
        runtime = DemoStateMachine(
            config,
            services=RobotComponents(FakeBase(), FakeArm(), depth, detector, FakeDetector("bin_tag")),
        )
        frame = depth.read_frame()
        detection = detector.detect(frame)
        required = runtime._incidental_can_confirmation_frames()
        self.assertEqual(required, 2)
        runtime._map_can_detections(frame, [detection], confirmation_frames=required)
        self.assertEqual(len(runtime.vague_map.known_cans), 0)
        runtime._map_can_detections(frame, [detection], confirmation_frames=required)
        self.assertEqual(len(runtime.vague_map.known_cans), 1)

    def test_bin_final_verify_resets_map_pose(self):
        config = load_config(overrides={
            "navigation": {"bin": {
                "align": {"target_error_x_norm": 0.0},
                "side_docking": {"experimental": {"enabled": False}},
            }},
        })
        runtime = DemoStateMachine(
            config,
            services=RobotComponents(FakeBase(), FakeArm(), FakeDepth(), FakeDetector("can"), FakeDetector("bin_tag")),
        )
        runtime.context.target_type = TargetType.BIN
        runtime.context.grabbed = True
        runtime.vague_map.robot_pose.x_m = 0.8
        runtime.vague_map.robot_pose.y_m = 0.8
        runtime.state = MissionState.FINAL_VERIFY
        runtime.context.begin_state(runtime.state)
        runtime.context.state_data["stable_frames"] = 1
        outcome = runtime.step_once()
        self.assertEqual(outcome.event, MissionEvent.TARGET_STABLE)
        self.assertEqual(runtime.vague_map.robot_pose.as_dict(), runtime.vague_map.bin_docking_pose.as_dict())

    def test_exp2_release_opens_only_gripper_then_resets_map_after_safe_home(self):
        events = []
        config = load_config(overrides={
            "arm": {"poses": {
                "side_view_bin_insert": {"pause_seconds": 0.0},
                "release": {"pause_seconds": 0.0},
                "safe_home": {"pause_seconds": 0.0},
            }},
        })
        base = FakeBase(events)
        arm = FakeArm(events)
        runtime = DemoStateMachine(
            config,
            services=RobotComponents(base, arm, FakeDepth(), FakeDetector("can"), FakeDetector("bin_tag")),
        )
        runtime.context.target_type = TargetType.BIN
        runtime.context.grabbed = True
        runtime.exp2_side_docked = True
        runtime.side_docking.arm_servo_angle = 70.0
        runtime.vague_map.robot_pose.x_m = 0.8
        runtime.vague_map.robot_pose.y_m = 0.8

        outcome = runtime._run_release_sequence()

        self.assertEqual(outcome.event, MissionEvent.FINALIZED)
        self.assertEqual(runtime.vague_map.robot_pose.as_dict(), runtime.vague_map.bin_side_docking_pose.as_dict())
        self.assertFalse(runtime.context.grabbed)
        insert_calls = [pose for name, pose in arm.poses if name == "side_view_bin_insert"]
        self.assertEqual(insert_calls, [{}])
        release_calls = [pose for name, pose in arm.poses if name == "release"]
        self.assertEqual(release_calls, [{"s4": 0}])
        self.assertEqual(base.commands, [("stop",)])
        self.assertNotIn("exp2_reset_heading", [command[0] for command in base.commands])
        self.assertLess(
            events.index(("arm_settled", "side_view_bin_insert")),
            events.index(("arm_pose", "release")),
        )
        self.assertLess(
            events.index(("arm_settled", "release")),
            events.index(("arm_pose", "safe_home")),
        )

    def test_side_tag_localization_uses_fixed_side_camera_heading(self):
        runtime = DemoStateMachine(
            load_config(overrides={
                "camera": {"calibration_yaml": "synthetic.yaml"},
                "vague_map": {"bin_tag_localization": {"enabled": True}},
            }),
            services=RobotComponents(
                FakeBase(), FakeArm(), FakeDepth(), FakeDetector("can"), FakeDetector("bin_tag")
            ),
        )
        captured = {}
        expected = runtime.vague_map.bin_side_docking_pose

        def localize(observation, camera_mount):
            captured.update(camera_mount)
            return expected

        runtime.vague_map.localize_robot_from_bin_tag = localize
        observation = {"found": True, "distance": 0.2, "pose": {"synthetic": True}}
        self.assertTrue(runtime._localize_from_bin_tag(observation, "side", expected))
        configured_heading = runtime.config.get(
            "vague_map.bin_tag_localization.camera_mounts.side.heading_rad"
        )
        self.assertAlmostEqual(captured["heading_rad"], configured_heading)

    def test_exp2_preserves_accepted_tag_localized_pose_after_release(self):
        class PnPBinDetector(FakeDetector):
            def detect(self, frame):
                observation = super(PnPBinDetector, self).detect(frame)
                observation.update({
                    "distance": (0.9 ** 2 + 0.1 ** 2) ** 0.5,
                    "pose": {
                        "x": -0.1,
                        "y": 0.0,
                        "z": 0.9,
                        "rotation_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                    },
                })
                return observation

        config = load_config(overrides={
            "camera": {"calibration_yaml": "synthetic.yaml"},
            "vague_map": {
                "bin_marker_position": {"x_m": 0.0, "y_m": 0.0, "heading_rad": 0.0},
                "bin_side_docking_pose": {"x_m": 0.0, "y_m": -1.0, "heading_rad": 0.0},
                "bin_tag_localization": {
                    "enabled": True,
                    "camera_mounts": {
                        "side": {"x_m": 0.0, "y_m": 0.0, "heading_rad": 0.0},
                    },
                },
            },
            "arm": {"poses": {
                "release": {"pause_seconds": 0.0},
                "safe_home": {"pause_seconds": 0.0},
            }},
        })
        runtime = DemoStateMachine(
            config,
            services=RobotComponents(
                FakeBase(), FakeArm(), FakeDepth(), FakeDetector("can"), PnPBinDetector("bin_tag")
            ),
        )
        runtime.context.target_type = TargetType.BIN
        runtime.context.grabbed = True
        runtime.state = MissionState.BIN_SIDE_DOCKING
        runtime.context.begin_state(runtime.state)
        runtime.context.state_data["entry_complete"] = True
        runtime.context.state_data["stable_frames"] = 4

        outcome = runtime.step_once()
        self.assertEqual(outcome.event, MissionEvent.TARGET_STABLE)
        self.assertTrue(runtime.bin_pose_corrected_from_tag)
        self.assertAlmostEqual(runtime.vague_map.robot_pose.x_m, 0.1)
        self.assertAlmostEqual(runtime.vague_map.robot_pose.y_m, -0.9)

        runtime.step_once()
        self.assertEqual(runtime.state, MissionState.PLANNING)
        self.assertAlmostEqual(runtime.vague_map.robot_pose.x_m, 0.1)
        self.assertAlmostEqual(runtime.vague_map.robot_pose.y_m, -0.9)

    def test_planning_prefers_nearest_mapped_can_after_release(self):
        from demo_core.vague_map import Point2D

        config = load_config()
        runtime = DemoStateMachine(
            config,
            services=RobotComponents(FakeBase(), FakeArm(), FakeDepth(), MissingDetector("can"), FakeDetector("bin_tag")),
        )
        far_can = runtime.vague_map.remember_can(Point2D(0.8, 0.0), 0.9)
        near_can = runtime.vague_map.remember_can(Point2D(0.2, 0.0), 0.7)
        outcome = runtime._handle_planning_state()
        self.assertEqual(outcome.event, MissionEvent.MAP_TARGET_AVAILABLE)
        self.assertEqual(runtime.vague_map.selected_can_id, near_can.can_id)
        self.assertNotEqual(far_can.can_id, near_can.can_id)

    def test_zero_pickup_limit_without_cached_can_restarts_searching(self):
        config = load_config(overrides={"runtime": {"max_pickups": 0}})
        runtime = DemoStateMachine(
            config,
            services=RobotComponents(FakeBase(), FakeArm(), FakeDepth(), MissingDetector("can"), FakeDetector("bin_tag")),
        )
        outcome = runtime._handle_planning_state()
        self.assertEqual(outcome.event, MissionEvent.TARGET_REQUIRED)


if __name__ == "__main__":
    unittest.main()
