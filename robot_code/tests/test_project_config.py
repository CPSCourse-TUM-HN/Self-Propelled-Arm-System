import os
import copy
import json
import math
import tempfile
import unittest

from demo_core.config import (
    ASSETS_DIR,
    APRILTAG_IMAGE_PATH,
    BASE_DIR,
    CAN_MODEL_DIR,
    CONFIG_PATH,
    DIAGNOSTIC_OUTPUT_DIR,
    EMPIRICAL_PARAMETERS_PATH,
    PREDEFINED_ROUTINES_PATH,
    LOG_DIR,
    PROJECT_PATHS,
    TESTS_DIR,
    BASE_SPEED_REFERENCE_PATHS,
    load_config,
    load_empirical_parameters,
)


class ProjectConfigTest(unittest.TestCase):
    def test_delivery_paths_are_configured_and_rooted(self):
        self.assertTrue(CONFIG_PATH.is_file())
        self.assertEqual(PROJECT_PATHS["project_root"], ".")
        self.assertEqual(EMPIRICAL_PARAMETERS_PATH, BASE_DIR / "empirical_parameters.json")
        self.assertEqual(PREDEFINED_ROUTINES_PATH, BASE_DIR / "predefined_routines.json")
        self.assertTrue(PREDEFINED_ROUTINES_PATH.is_file())
        self.assertEqual(ASSETS_DIR, BASE_DIR / "assets")
        self.assertEqual(TESTS_DIR, BASE_DIR / "tests")
        self.assertEqual(CAN_MODEL_DIR, ASSETS_DIR / "models" / "detectnet_native_can")
        self.assertEqual(APRILTAG_IMAGE_PATH, ASSETS_DIR / "bin_apriltag_36h11_id_0.png")
        self.assertEqual(LOG_DIR, BASE_DIR / "logs")
        self.assertEqual(DIAGNOSTIC_OUTPUT_DIR, BASE_DIR / "diagnostic_outputs")
        self.assertTrue(os.path.isdir(str(ASSETS_DIR)))
        self.assertTrue(APRILTAG_IMAGE_PATH.is_file())
        self.assertTrue((CAN_MODEL_DIR / "can_ssd_mobilenet_v1.onnx").is_file())
        self.assertNotIn("legacy_params", PROJECT_PATHS)

    def test_merge_priority_is_parameters_then_config_then_overrides(self):
        with open(str(EMPIRICAL_PARAMETERS_PATH), "r") as stream:
            parameters = json.load(stream)
        with open(str(CONFIG_PATH), "r") as stream:
            runtime_config = json.load(stream)
        parameters = copy.deepcopy(parameters)
        runtime_config = copy.deepcopy(runtime_config)
        parameters.setdefault("runtime", {}).setdefault("dry_run", {})["arm"] = True
        runtime_config["runtime"]["dry_run"]["arm"] = False
        runtime_config["paths"]["project_root"] = str(BASE_DIR)
        with tempfile.TemporaryDirectory() as directory:
            parameters_path = os.path.join(directory, "parameters.json")
            config_path = os.path.join(directory, "config.json")
            with open(parameters_path, "w") as stream:
                json.dump(parameters, stream)
            with open(config_path, "w") as stream:
                json.dump(runtime_config, stream)
            self.assertFalse(load_config(parameters_path, config_path).get("runtime.dry_run.arm"))
            merged = load_config(
                parameters_path,
                config_path,
                overrides={"runtime": {"dry_run": {"arm": True}}},
            )
            self.assertTrue(merged.get("runtime.dry_run.arm"))

    def test_vague_map_is_enabled_and_single_pickup_is_default(self):
        config = load_config()
        self.assertFalse(config.get("camera.depth_rectification_enabled"))
        self.assertFalse(config.get("detectors.can.rectification_enabled"))
        self.assertEqual(config.get("camera.calibration_yaml"), "calibration_320.yaml")
        self.assertIsNone(config.get("detectors.bin.calibration_yaml"))
        self.assertEqual(
            os.path.dirname(config.get("detectors.can.model_path")),
            os.path.dirname(config.get("detectors.can.labels_path")),
        )
        self.assertTrue(config.get("vague_map.enabled"))
        self.assertIsNone(config.get("vague_map.incidental_can_min_center_y_norm"))
        self.assertEqual(config.get("navigation.can.near_align.required_stable_frames"), 2)
        self.assertEqual(config.get("runtime.max_pickups"), 1)
        self.assertTrue(config.get("vague_map.bin_tag_localization.enabled"))

    def test_rectification_configuration_is_validated(self):
        with self.assertRaisesRegex(ValueError, "rectification_alpha"):
            load_config(overrides={"camera": {"rectification_alpha": 1.1}})
        with self.assertRaisesRegex(ValueError, "camera.calibration_yaml"):
            load_config(overrides={"camera": {
                "calibration_yaml": "",
                "depth_rectification_enabled": True,
            }})

    def test_bin_tag_localization_requires_camera_calibration(self):
        with self.assertRaises(ValueError):
            load_config(overrides={
                "camera": {"calibration_yaml": ""},
                "vague_map": {"bin_tag_localization": {"enabled": True}},
            })

        config = load_config(overrides={
            "camera": {"calibration_yaml": "synthetic.yaml"},
            "vague_map": {"bin_tag_localization": {"enabled": True}},
        })
        self.assertTrue(config.validate())

    def test_map_disabled_requires_positive_pickup_limit(self):
        with self.assertRaises(ValueError):
            load_config(overrides={
                "vague_map": {"enabled": False},
                "runtime": {"max_pickups": 0},
            })

    def test_base_motion_speeds_use_named_profiles(self):
        empirical = load_empirical_parameters()
        profiles = empirical["base_motion_speed_profiles"]
        self.assertEqual(profiles["linear"], {"fast": 0.6, "slow": 0.2})
        self.assertEqual(profiles["turn"], {"fast": 0.3, "slow": 0.15})

        config = load_config()
        self.assertAlmostEqual(config.get("navigation.bin.align.target_error_x_norm"), -0.16)
        with self.assertRaises(ValueError):
            load_config(overrides={
                "navigation": {"bin": {"align": {"target_error_x_norm": 1.1}}},
            })
        self.assertEqual(config.get("navigation.bin.approach.stop.mode"), "pnp_z")
        self.assertAlmostEqual(config.get("navigation.bin.approach.stop.pnp_z_threshold_m"), 0.36)
        with self.assertRaises(ValueError):
            load_config(overrides={"navigation": {"bin": {"approach": {"stop": {
                "mode": "pnp_z", "pnp_z_threshold_m": 0.0,
            }}}}})
        for path, motion_type in BASE_SPEED_REFERENCE_PATHS.items():
            raw = empirical
            for key in path.split("."):
                raw = raw[key]
            self.assertIn(raw, (motion_type + ".fast", motion_type + ".slow"))
            self.assertNotIn("reserve", raw)
            level = raw.split(".")[1]
            self.assertEqual(config.get(path), profiles[motion_type][level])

    def test_linear_response_contains_measured_slow_and_fast_samples(self):
        odometry = load_empirical_parameters()["vague_map"]["odometry"]
        samples = {sample["speed"]: sample for sample in odometry["linear_response_samples"]}
        self.assertEqual(samples[0.2], {"speed": 0.2, "seconds": 4.0, "distance_m": 0.13})
        self.assertEqual(samples[0.6], {"speed": 0.6, "seconds": 4.0, "distance_m": 0.415})

    def test_base_motion_command_scales_are_neutral_and_positive(self):
        empirical = load_empirical_parameters()
        self.assertEqual(
            empirical["base_motion_command_scales"],
            {"forward": 1.0, "backward": 1.0, "left": 1.0, "right": 1.0},
        )
        with self.assertRaises(ValueError):
            load_config(overrides={"base_motion_command_scales": {"left": 0.0}})
        with self.assertRaises(ValueError):
            load_config(overrides={"base_motion_command_scales": {"left": float("nan")}})

    def test_turn_response_contains_measured_fixed_speeds(self):
        empirical = load_empirical_parameters()
        samples = {row["speed"]: row for row in empirical["base_turn_response"]["samples"]}
        self.assertEqual(samples[0.15]["left_seconds"], 10.4)
        self.assertEqual(samples[0.15]["right_seconds"], 9.0)
        self.assertEqual(samples[0.3]["left_seconds"], 3.6)
        self.assertEqual(samples[0.3]["right_seconds"], 3.4)
        with self.assertRaises(ValueError):
            load_config(overrides={"base_turn_response": {"samples": [{
                "speed": 0.15, "left_seconds": 0.0, "right_seconds": 9.0,
            }]}})

    def test_redundant_motion_parameters_are_absent(self):
        empirical = load_empirical_parameters()
        self.assertNotIn("reserve", empirical["base_motion_speed_profiles"])
        self.assertNotIn("gripper_open", empirical["arm"])
        self.assertNotIn("gripper_close", empirical["arm"])
        self.assertNotIn("patrol_waypoints", empirical["vague_map"])
        self.assertNotIn("waypoint_tolerance_m", empirical["vague_map"]["navigation"])

        for target_name in ("can", "bin"):
            navigation = empirical["navigation"][target_name]
            self.assertEqual(navigation["search"]["routine"], 0)
            if target_name == "bin":
                self.assertEqual(navigation["search"]["map_arrival_routine"], 4)
            else:
                self.assertNotIn("map_arrival_routine", navigation["search"])
            self.assertNotIn("pulse_seconds", navigation["search"])

            self.assertNotIn("pulse_seconds", navigation["align"])
            self.assertNotIn("pulse_seconds", navigation["approach"])
            self.assertNotIn("steering_pulse_seconds", navigation["approach"])
            self.assertNotIn("min_pulses", navigation["approach"])
        self.assertNotIn("pulse_seconds", empirical["navigation"]["can"]["near_align"])

        exp2 = empirical["navigation"]["bin"]["side_docking"]["experimental"]
        self.assertNotIn("drive_pulse_seconds", exp2)
        self.assertNotIn("turn_pulse_seconds", exp2)
        self.assertNotIn("perspective_tolerance", exp2)
        self.assertNotIn("perspective_positive_turn_direction", exp2)
        self.assertNotIn("turn_speed", exp2)
        self.assertNotIn("supportive_turn_direction", exp2)
        self.assertNotIn("supportive_turn_speed", exp2)
        self.assertNotIn("supportive_turn_seconds", exp2)
        self.assertNotIn("corner_angle_tolerance_deg", exp2)
        self.assertAlmostEqual(exp2["yaw_tolerance_rad"], 0.128131701)
        self.assertAlmostEqual(exp2["target_yaw_rad"], 0.0)
        self.assertEqual(exp2["yaw_positive_turn_direction"], "right")
        self.assertEqual(exp2["side_entry_base_turn_direction"], "left")
        self.assertEqual(exp2["yaw_turn_speed"], "turn.slow")
        self.assertNotIn("yaw_arm_compensation_enabled", exp2)
        self.assertNotIn("yaw_arm_servo_positive_base_direction", exp2)
        self.assertNotIn("yaw_arm_max_rotation_rad", exp2)
        self.assertIsNone(load_config().get("detectors.can.enabled"))

    def test_release_poses_only_open_gripper_after_bin_insert(self):
        poses = load_empirical_parameters()["arm"]["poses"]
        insert = poses["side_view_bin_insert"]["angles"]
        release = poses["release"]["angles"]
        expected_fixed = {"s1": 70, "s2": 80, "s3": 55, "s5": 0}
        for pose in (insert, release):
            self.assertEqual({key: pose[key] for key in expected_fixed}, expected_fixed)
        self.assertEqual(insert["s4"], -55)
        self.assertEqual(release["s4"], 0)

    def test_removed_keys_fail_explicitly(self):
        removed_overrides = (
            {"arm": {"verify_enabled": False}},
            {"camera": {"target_depth_roi": [0.4, 0.4, 0.6, 0.6]}},
            {"detectors": {"bin": {"calibration_workarounds": {"enabled": False}}}},
            {"detectors": {"bin": {"calibration_yaml": "old.yaml"}}},
            {"navigation": {"can": {"approach": {"final_verify_frames": 2}}}},
            {"navigation": {"can": {"approach": {"min_pulses": 0}}}},
            {"navigation": {"bin": {"approach": {"min_pulses": 0}}}},
            {"navigation": {"bin": {"side_docking": {"experimental": {
                "corner_angle_tolerance_deg": 90.0,
            }}}}},
            {"navigation": {"bin": {"side_docking": {"experimental": {
                "standoff_correction_enabled": False,
            }}}}},
        )
        for overrides in removed_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "removed config key"):
                    load_config(overrides=overrides)

    def test_base_motion_speed_rejects_wrong_profile_type(self):
        with self.assertRaises(ValueError):
            load_config(overrides={
                "navigation": {"can": {"search": {"routine": 999}}},
            })


if __name__ == "__main__":
    unittest.main()
