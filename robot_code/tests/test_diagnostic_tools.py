from __future__ import print_function

import csv
import json
import os
import tempfile
import unittest

import numpy as np

from demo_core.config import load_config
from demo_core.fsm_types import MissionEvent, MissionState
from demo_core.perception import AprilTagBinDetector
from tuning_tools.base_speed_time_calibration import annotate_measurement, turn_response_metrics
from tuning_tools.diagnostic_tools import (
    CandidateCountTracker,
    candidate_bbox_metrics,
    decode_scscl_status,
    pose_from_tag_corners,
    robust_stats,
    summarize_pose_samples,
)
from tuning_tools.bin_docking_tuning import BinDockingTuningPanel, critical_parameter_snapshot
from tuning_tools.vague_map_tester import (
    build_test_config,
    calculate_forward_factor,
    first_motion,
    load_test_parameters,
    preview,
    VagueMapTestSession,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = (
    "tests/full_demo_integration_test.ipynb",
    "tuning_tools/camera_depthnet_interrupt_test.ipynb",
    "tuning_tools/camera_network_diagnostics.ipynb",
    "tuning_tools/bin_docking_tuning.ipynb",
    "tuning_tools/base_motion_performance.ipynb",
    "tuning_tools/base_speed_time_calibration.ipynb",
    "tuning_tools/arm_grasp_telemetry_test.ipynb",
    "tuning_tools/dual_obstacle_detection_test.ipynb",
    "tuning_tools/dual_can_vague_map_recording_test.ipynb",
    "tuning_tools/camera_intrinsic_calibration.ipynb",
    "tuning_tools/apriltag_pnp_stability_test.ipynb",
    "tuning_tools/vague_map_tester.ipynb",
)


class FakeAnalysis(object):
    def __init__(self):
        self.mask = np.zeros((10, 20), dtype=np.uint8)
        self.mask[2:8, 2:8] = 255
        self.mask[2:8, 12:18] = 255
        self.candidate_bboxes = [(2, 2, 6, 6), (12, 2, 6, 6)]


class DiagnosticToolsTest(unittest.TestCase):
    def test_bin_docking_panel_uses_json_parameters_and_defaults_to_safe_stop(self):
        config = load_config()
        parameters = critical_parameter_snapshot(config)
        self.assertAlmostEqual(parameters["front_align_target_error_x_norm"], -0.16)
        self.assertAlmostEqual(parameters["front_pnp_z_stop_threshold_m"], 0.36)
        self.assertAlmostEqual(parameters["target_yaw_deg"], 0.0)
        self.assertAlmostEqual(parameters["yaw_tolerance_deg"], 7.3414056891)

        panel = BinDockingTuningPanel()
        panel.camera_real.value = False
        panel.base_real.value = False
        panel.arm_real.value = False
        self.assertEqual(panel.start_stage.value, "align")
        self.assertFalse(panel.release_after_correction.value)
        self.assertEqual(
            panel.completion_result(MissionState.BIN_SIDE_DOCKING, MissionState.FINALIZING),
            "CORRECTION_COMPLETE",
        )
        panel.release_after_correction.value = True
        self.assertIsNone(
            panel.completion_result(MissionState.BIN_SIDE_DOCKING, MissionState.FINALIZING)
        )
        self.assertEqual(
            panel.completion_result(MissionState.FINALIZING, MissionState.PLANNING),
            "RELEASE_COMPLETE",
        )

        path = os.path.join(PROJECT_ROOT, "tuning_tools/bin_docking_tuning.ipynb")
        with open(path, "r") as stream:
            notebook = json.load(stream)
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("BinDockingTuningPanel", source)
        self.assertIn("release_after_correction", source)
        self.assertIn("start_stage=align", source)

    def test_bin_docking_helper_uses_thread_routing_and_reuses_live_camera(self):
        path = os.path.join(PROJECT_ROOT, "tuning_tools/bin_docking_tuning.ipynb")
        with open(path, "r") as stream:
            notebook = json.load(stream)
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        helper_path = os.path.join(PROJECT_ROOT, "tuning_tools/bin_docking_tuning.py")
        with open(helper_path, "r") as stream:
            helper_source = stream.read()
        self.assertIn("rt.navigator.frame_observer = None", helper_source)
        self.assertIn("class ThreadOutputRouter", helper_source)
        self.assertIn("release_after_correction", helper_source)
        self.assertNotIn("contextlib.redirect_stdout", helper_source)

        integration_path = os.path.join(PROJECT_ROOT, "tests/full_demo_integration_test.ipynb")
        with open(integration_path, "r") as stream:
            integration = json.load(stream)
        integration_source = "\n".join(
            "".join(cell.get("source", [])) for cell in integration["cells"]
        )
        self.assertIn("class ThreadOutputRouter", integration_source)
        self.assertIn("with routed_output(stdout_tee, stderr_tee):", integration_source)
        self.assertNotIn("contextlib.redirect_stdout", integration_source)

    def test_bin_docking_dry_run_supports_stop_and_release_modes(self):
        for release_after_correction in (False, True):
            with self.subTest(release_after_correction=release_after_correction):
                panel = BinDockingTuningPanel()
                panel.camera_real.value = False
                panel.base_real.value = False
                panel.arm_real.value = False
                panel.live.value = False
                panel.record.value = False
                panel.release_after_correction.value = release_after_correction
                runtime, _ = panel._prepare()
                exp2 = runtime.config.get("navigation.bin.side_docking.experimental")
                exp2.update({
                    "post_entry_camera_settle_seconds": 0.0,
                    "post_entry_camera_discard_frames": 0,
                    "post_correction_camera_settle_seconds": 0.0,
                    "stable_frames": 1,
                })
                original_detect = runtime.services.bin_detector.detect

                def stable_detection(frame):
                    result = original_detect(frame)
                    result["error_x"] = (
                        0.0 if runtime.state == MissionState.BIN_SIDE_DOCKING else -0.16
                    )
                    return result

                runtime.services.bin_detector.detect = stable_detection
                completion = None
                try:
                    for _ in range(30):
                        previous = runtime.state
                        runtime.step_once()
                        completion = panel.completion_result(previous, runtime.state)
                        if completion is not None:
                            break
                    expected = "RELEASE_COMPLETE" if release_after_correction else "CORRECTION_COMPLETE"
                    self.assertEqual(completion, expected)
                    self.assertIsNone(runtime.services.base.command_snapshot())
                    self.assertEqual(runtime.context.grabbed, not release_after_correction)
                finally:
                    runtime.stop_all()

    def test_scscl_status_decodes_signed_fields_and_voltage(self):
        status = decode_scscl_status({
            "position": 512,
            "speed": (1 << 15) | 12,
            "load": (1 << 10) | 34,
            "voltage": 65,
            "temperature": 21,
            "moving": 1,
            "current": (1 << 15) | 56,
        })
        self.assertEqual(status["speed"], -12)
        self.assertEqual(status["load"], -34)
        self.assertEqual(status["current"], -56)
        self.assertEqual(status["voltage_v"], 6.5)
        self.assertTrue(status["moving"])

    def test_candidate_metrics_and_count_rates(self):
        metrics = candidate_bbox_metrics(FakeAnalysis(), np.full((10, 20), 1.25, dtype=np.float32))
        self.assertEqual(len(metrics), 2)
        self.assertAlmostEqual(metrics[0]["median_depth"], 1.25)
        tracker = CandidateCountTracker(2)
        for count in (2, 1, 3, 2):
            tracker.update(count)
        self.assertEqual(
            tracker.summary(),
            {
                "frames": 4,
                "expected_count_rate": 0.5,
                "merge_rate": 0.25,
                "false_split_rate": 0.25,
                "identity_stability_rate": 0.0,
            },
        )
        stable = CandidateCountTracker(2)
        stable.update_candidates(metrics)
        stable.update_candidates(metrics)
        self.assertEqual(stable.summary()["identity_stability_rate"], 1.0)

    def test_robust_pose_statistics(self):
        stats = robust_stats([1.0, 2.0, 100.0])
        self.assertEqual(stats["median"], 2.0)
        self.assertEqual(stats["mad"], 1.0)
        samples = [{field: 1.0 for field in (
            "x_m", "y_m", "z_m", "perpendicular_distance_m", "distance_m",
            "yaw_deg", "pitch_deg", "roll_deg",
        )}]
        self.assertEqual(summarize_pose_samples(samples, 2)["detection_rate"], 0.5)

    def test_pose_recovery_from_synthetic_square(self):
        import cv2

        matrix = np.asarray([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        half = 0.04
        object_points = np.asarray([[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]], dtype=np.float32)
        corners, _ = cv2.projectPoints(object_points, np.zeros(3), np.asarray([0.0, 0.0, 0.5]), matrix, np.zeros(5))
        pose = pose_from_tag_corners(corners.reshape(4, 2), matrix, np.zeros(5), 0.08)
        self.assertAlmostEqual(pose["z_m"], 0.5, places=3)
        self.assertAlmostEqual(pose["distance_m"], 0.5, places=3)

    def test_runtime_tag_pose_keeps_rotation_for_map_localization(self):
        import cv2

        matrix = np.asarray([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        half = 0.04
        object_points = np.asarray([[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]], dtype=np.float32)
        corners, _ = cv2.projectPoints(object_points, np.zeros(3), np.asarray([0.0, 0.0, 0.5]), matrix, np.zeros(5))
        detector = AprilTagBinDetector(load_config())
        detector.camera_matrix = matrix
        detector.dist_coeffs = np.zeros(5)
        result = detector._estimate_pose(corners.reshape(4, 2))
        self.assertEqual(len(result["pose"]["rotation_matrix"]), 3)
        self.assertAlmostEqual(result["pose"]["rotation_matrix"][2][2], 1.0, places=5)

    def test_turn_metrics_and_csv_annotation(self):
        metrics = turn_response_metrics(60.0, 0.2, 1.0)
        self.assertEqual(metrics["turn_deg_per_speed_second"], 300.0)
        self.assertEqual(metrics["recommended_90_seconds"], 1.5)
        fields = [
            "run_id", "motion", "effective_speed", "seconds", "measurement_kind", "measurement_value",
            "turn_deg_per_speed_second", "recommended_90_seconds", "suggested_direction_scale",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "calibration.csv")
            with open(path, "w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"run_id": "left", "motion": "left", "effective_speed": 0.2, "seconds": 1.0})
                writer.writerow({"run_id": "right", "motion": "right", "effective_speed": 0.2, "seconds": 1.0})
            annotate_measurement(path, "left", 60.0)
            row = annotate_measurement(path, "right", 50.0)
            self.assertAlmostEqual(float(row["recommended_90_seconds"]), 1.8)
            self.assertTrue(row["suggested_direction_scale"])

    def test_vague_map_tester_direction_and_forward_factor(self):
        destination = {"x_m": 0.0, "y_m": 0.6}
        left_start = {"x_m": -0.5, "y_m": 0.0, "heading_rad": 0.0}
        right_start = {"x_m": 0.5, "y_m": 0.0, "heading_rad": 0.0}
        self.assertEqual(first_motion(left_start, destination, 0.12, 0.2)["motion"], "right")
        self.assertEqual(first_motion(right_start, destination, 0.12, 0.2)["motion"], "left")
        self.assertAlmostEqual(calculate_forward_factor(0.48, 0.6, 1.0), 0.8)

    def test_vague_map_tester_parameters_build_valid_override(self):
        path = os.path.join(PROJECT_ROOT, "tuning_tools", "vague_map_test_parameters.json")
        data = load_test_parameters(path)
        config = build_test_config(data)
        prediction = preview(data, config)
        self.assertEqual(config.get("vague_map.initial_pose.x_m"), -0.5)
        self.assertEqual(config.get("vague_map.navigation.forward_speed"), 0.6)
        self.assertEqual(prediction["motion"], "right")
        self.assertTrue(prediction["matches_expectation"])

    def test_vague_map_tester_planning_overrides_search(self):
        session = VagueMapTestSession()
        session.reload()
        runtime = session._make_runtime(force_tag_missing=True)
        outcome = runtime.step_once()
        self.assertEqual(outcome.event, MissionEvent.MAP_TARGET_AVAILABLE)
        self.assertEqual(runtime.state, MissionState.MAP_NAVIGATING)

    def test_changed_notebooks_are_clean_valid_json(self):
        for relative in NOTEBOOKS:
            path = os.path.join(PROJECT_ROOT, relative)
            with open(path, "r") as stream:
                notebook = json.load(stream)
            self.assertEqual(notebook["nbformat"], 4, relative)
            for cell in notebook["cells"]:
                if cell.get("cell_type") == "code":
                    self.assertIsNone(cell.get("execution_count"), relative)
                    self.assertEqual(cell.get("outputs", []), [], relative)
                    compile("".join(cell.get("source", [])), relative, "exec")
            source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
            self.assertIn("config.json", source, relative)


if __name__ == "__main__":
    unittest.main()
