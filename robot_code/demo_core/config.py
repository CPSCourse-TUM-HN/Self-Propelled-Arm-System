from __future__ import print_function

import copy
import json
import math
import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PACKAGE_ROOT / "config.json"


BASE_SPEED_REFERENCE_PATHS = {
    "navigation.can.align.speed": "turn",
    "navigation.can.approach.speed": "linear",
    "navigation.can.approach.steering_speed": "turn",
    "navigation.can.near_align.speed": "turn",
    "navigation.bin.align.speed": "turn",
    "navigation.bin.approach.speed": "linear",
    "navigation.bin.approach.steering_speed": "turn",
    "navigation.bin.side_docking.experimental.drive_speed": "linear",
    "navigation.bin.side_docking.experimental.yaw_turn_speed": "turn",
    "vague_map.navigation.turn_speed": "turn",
    "vague_map.navigation.forward_speed": "linear",
    "avoidance.turn_speed": "turn",
    "avoidance.forward_speed": "linear",
    "arm.push.speed": "linear",
}

REMOVED_CONFIG_PATHS = (
    "arm.gripper_speed",
    "arm.poses.side_view_release_low",
    "arm.verify_depth",
    "arm.verify_enabled",
    "camera.target_depth_roi",
    "detectors.bin.calibration_yaml",
    "navigation.can.approach.min_pulses",
    "detectors.bin.calibration_workarounds",
    "navigation.can.approach.experimental_continuous",
    "navigation.can.approach.final_verify_frames",
    "navigation.bin.approach.final_verify_frames",
    "navigation.bin.approach.min_pulses",
    "navigation.bin.side_docking.experimental.corner_angle_tolerance_deg",
    "navigation.bin.side_docking.experimental.final_yaw_correction_limit_enabled",
    "navigation.bin.side_docking.experimental.final_yaw_correction_limit_rad",
    "navigation.bin.side_docking.experimental.height_tolerance_norm",
    "navigation.bin.side_docking.experimental.standoff_correction_enabled",
    "navigation.bin.side_docking.experimental.standoff_drive_seconds",
    "navigation.bin.side_docking.experimental.standoff_drive_speed",
    "navigation.bin.side_docking.experimental.standoff_toward_turn_direction",
    "navigation.bin.side_docking.experimental.standoff_turn_seconds",
    "navigation.bin.side_docking.experimental.standoff_turn_speed",
    "navigation.bin.side_docking.experimental.target_height_norm",
    "navigation.bin.side_docking.experimental.yaw_zero_reference_rad",
)


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _require(mapping, path):
    value = mapping
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError("missing config key: {}".format(path))
        value = value[key]
    return value


def _has_path(mapping, path):
    value = mapping
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return True


def _read_json(path):
    with open(str(path), "r") as stream:
        return json.load(stream)


def _set_path(mapping, path, value):
    target = mapping
    keys = path.split(".")
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


def _resolve_base_speed_references(data):
    profiles = _require(data, "base_motion_speed_profiles")
    for motion_type in ("linear", "turn"):
        values = _require(profiles, motion_type)
        for level in ("fast", "slow"):
            if float(_require(values, level)) <= 0.0:
                raise ValueError("base_motion_speed_profiles.{}.{} must be positive".format(motion_type, level))
    for path, expected_type in BASE_SPEED_REFERENCE_PATHS.items():
        reference = _require(data, path)
        if not isinstance(reference, str):
            raise ValueError("{} must use a symbolic base speed reference".format(path))
        parts = reference.split(".")
        if len(parts) != 2 or parts[0] != expected_type or parts[1] not in ("fast", "slow"):
            raise ValueError(
                "{} must reference {}.fast or {}.slow; got {}".format(
                    path, expected_type, expected_type, reference
                )
            )
        _set_path(data, path, float(profiles[parts[0]][parts[1]]))
    return data


def _runtime_paths(config_path=None):
    config_path = Path(config_path or CONFIG_PATH).resolve()
    runtime_config = _read_json(config_path)
    paths = runtime_config.get("paths", {})
    root_value = Path(paths.get("project_root", "."))
    project_root = root_value if root_value.is_absolute() else (config_path.parent / root_value).resolve()
    return config_path, runtime_config, paths, project_root


class DemoConfig(object):
    REQUIRED_PATHS = (
        "runtime.dry_run.base",
        "runtime.dry_run.arm",
        "runtime.dry_run.camera",
        "runtime.retry_limit",
        "paths.predefined_routines",
        "camera.width",
        "camera.height",
        "camera.calibration_yaml",
        "camera.depth_rectification_enabled",
        "detectors.can.model_path",
        "detectors.can.labels_path",
        "detectors.can.confidence_threshold",
        "detectors.bin.tag_id",
        "navigation.can.search",
        "navigation.can.align",
        "navigation.can.approach",
        "navigation.bin.search",
        "navigation.bin.align",
        "navigation.bin.approach",
        "arm.poses.safe_home",
        "arm.poses.arm_down",
        "arm.poses.grab",
        "arm.poses.carry",
        "arm.poses.release",
    )

    def __init__(self, data, config_path, parameters_path, project_root):
        self.data = data
        self.config_path = os.path.abspath(str(config_path))
        self.parameters_path = os.path.abspath(str(parameters_path))
        self.project_root = os.path.abspath(str(project_root))
        self.validate()

    def validate(self):
        for path in REMOVED_CONFIG_PATHS:
            if _has_path(self.data, path):
                raise ValueError("removed config key is no longer supported: {}".format(path))
        for path in self.REQUIRED_PATHS:
            _require(self.data, path)
        rectification_alpha = float(self.get("camera.rectification_alpha", 0.0))
        if not math.isfinite(rectification_alpha) or not 0.0 <= rectification_alpha <= 1.0:
            raise ValueError("camera.rectification_alpha must be finite and in [0, 1]")
        if bool(self.get("camera.depth_rectification_enabled", False)) or bool(
                self.get("detectors.can.rectification_enabled", False)):
            if not str(self.get("camera.calibration_yaml", "")).strip():
                raise ValueError("camera.calibration_yaml is required when rectification is enabled")
        command_scales = _require(self.data, "base_motion_command_scales")
        for direction in ("forward", "backward", "left", "right"):
            value = float(_require(command_scales, direction))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("base_motion_command_scales.{} must be positive".format(direction))
        turn_response = _require(self.data, "base_turn_response")
        reference_angle = float(_require(turn_response, "reference_angle_rad"))
        if not math.isfinite(reference_angle) or reference_angle <= 0.0:
            raise ValueError("base_turn_response.reference_angle_rad must be positive")
        samples = _require(turn_response, "samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("base_turn_response.samples must be a non-empty list")
        seen_speeds = set()
        for sample in samples:
            speed = float(_require(sample, "speed"))
            if not math.isfinite(speed) or speed <= 0.0 or speed > 1.0:
                raise ValueError("base_turn_response sample speed must be in (0.0, 1.0]")
            if speed in seen_speeds:
                raise ValueError("duplicate base_turn_response speed: {}".format(speed))
            seen_speeds.add(speed)
            for direction in ("left", "right"):
                seconds = float(_require(sample, direction + "_seconds"))
                if not math.isfinite(seconds) or seconds <= 0.0:
                    raise ValueError("base_turn_response {}_seconds must be positive".format(direction))
        for target_name in ("can", "bin"):
            target = self.data["navigation"][target_name]
            routine_id = _require(target, "search.routine")
            from .searching_routine import SearchingRoutineLibrary
            routine_library = SearchingRoutineLibrary(self.section("predefined_routines"), self)
            routine_library.get(routine_id)
            if target_name == "bin" and "map_arrival_routine" in target["search"]:
                routine_library.get(target["search"]["map_arrival_routine"])
            if int(target["align"]["max_steps"]) <= 0:
                raise ValueError("{}.align.max_steps must be positive".format(target_name))
            target_error_x = float(target["align"].get("target_error_x_norm", 0.0))
            if not math.isfinite(target_error_x) or not -1.0 <= target_error_x <= 1.0:
                raise ValueError(
                    "{}.align.target_error_x_norm must be finite and in [-1, 1]".format(target_name)
                )
            if int(target["align"].get("lost_motion_grace_frames", 0)) < 0:
                raise ValueError("{}.align.lost_motion_grace_frames must be non-negative".format(target_name))
            if float(target["approach"]["timeout_seconds"]) <= 0:
                raise ValueError("{}.approach.timeout_seconds must be positive".format(target_name))
            stop = _require(target, "approach.stop")
            stop_mode = str(_require(stop, "mode"))
            allowed_stop_modes = ("bbox_height", "depth", "bbox_height_and_depth")
            if target_name == "bin":
                allowed_stop_modes += ("pnp_z",)
            if stop_mode not in allowed_stop_modes:
                raise ValueError("{}.approach.stop.mode is invalid".format(target_name))
            legacy_threshold = stop.get("threshold")
            bbox_threshold = stop.get("bbox_height_threshold", legacy_threshold)
            depth_threshold = stop.get("depth_raw_threshold", legacy_threshold)
            if stop_mode in ("bbox_height", "bbox_height_and_depth"):
                if bbox_threshold is None or not 0.0 < float(bbox_threshold) <= 1.0:
                    raise ValueError(
                        "{}.approach.stop.bbox_height_threshold must be in (0, 1]".format(target_name)
                    )
            if stop_mode in ("depth", "bbox_height_and_depth"):
                if depth_threshold is None or not math.isfinite(float(depth_threshold)) or float(depth_threshold) <= 0.0:
                    raise ValueError(
                        "{}.approach.stop.depth_raw_threshold must be positive and finite".format(target_name)
                    )
            if stop_mode == "pnp_z":
                pnp_z_threshold = stop.get("pnp_z_threshold_m")
                if (
                    pnp_z_threshold is None
                    or not math.isfinite(float(pnp_z_threshold))
                    or float(pnp_z_threshold) <= 0.0
                ):
                    raise ValueError(
                        "{}.approach.stop.pnp_z_threshold_m must be positive and finite".format(target_name)
                    )
                calibration_yaml = str(self.get("camera.calibration_yaml", "")).strip()
                if not calibration_yaml:
                    raise ValueError("camera.calibration_yaml is required for bin PnP-z approach")
            if int(_require(target, "near_align.required_stable_frames")) <= 0:
                raise ValueError("{}.near_align.required_stable_frames must be positive".format(target_name))
        exp2 = self.get("navigation.bin.side_docking.experimental", {})
        if bool(exp2.get("enabled", False)):
            calibration_yaml = str(self.get("camera.calibration_yaml", "")).strip()
            target_yaw = float(exp2.get("target_yaw_rad", 0.0))
            if not math.isfinite(target_yaw):
                raise ValueError(
                    "navigation.bin.side_docking.experimental.target_yaw_rad must be finite"
                )
            if not calibration_yaml:
                raise ValueError("camera.calibration_yaml is required for side docking")
            for path in (
                "arm.poses.side_view_grabbing",
                "arm.poses.side_view_bin_insert",
                "vague_map.bin_side_docking_pose",
            ):
                _require(self.data, path)
            for key in (
                "drive_speed", "center_correction_seconds",
                "yaw_turn_speed", "yaw_correction_seconds", "yaw_tolerance_rad",
                "yaw_single_correction_max_seconds",
                "center_stage_entry_yaw_rad", "coarse_yaw_center_guard_norm",
                "timeout_seconds",
            ):
                if float(exp2.get(key, 0.0)) <= 0.0:
                    raise ValueError("navigation.bin.side_docking.experimental.{} must be positive".format(key))
            if exp2.get("correction_mode", "continuous") not in ("continuous", "pulse"):
                raise ValueError(
                    "navigation.bin.side_docking.experimental.correction_mode "
                    "must be continuous or pulse"
                )
            if float(exp2["center_stage_entry_yaw_rad"]) < float(exp2["yaw_tolerance_rad"]):
                raise ValueError(
                    "navigation.bin.side_docking.experimental.center_stage_entry_yaw_rad "
                    "must be at least yaw_tolerance_rad"
                )
            if float(exp2["coarse_yaw_center_guard_norm"]) > 0.5:
                raise ValueError(
                    "navigation.bin.side_docking.experimental.coarse_yaw_center_guard_norm "
                    "must not exceed 0.5"
                )
            for key in ("stable_frames", "lost_frame_limit", "max_steps"):
                if int(exp2.get(key, 0)) <= 0:
                    raise ValueError("navigation.bin.side_docking.experimental.{} must be positive".format(key))
            valid_directions = ("left", "right", "forward", "backward")
            for key in (
                "center_positive_drive_direction",
                "yaw_positive_turn_direction",
            ):
                if exp2.get(key) not in valid_directions:
                    raise ValueError("navigation.bin.side_docking.experimental.{} is invalid".format(key))
            if exp2.get("side_entry_base_turn_direction") not in ("left", "right"):
                raise ValueError(
                    "navigation.bin.side_docking.experimental.side_entry_base_turn_direction is invalid"
                )
            if float(exp2.get("post_entry_camera_settle_seconds", 0.0)) < 0.0:
                raise ValueError(
                    "navigation.bin.side_docking.experimental.post_entry_camera_settle_seconds must be non-negative"
                )
            if int(exp2.get("post_entry_camera_discard_frames", 0)) < 0:
                raise ValueError(
                    "navigation.bin.side_docking.experimental.post_entry_camera_discard_frames must be non-negative"
                )
            if float(exp2.get("post_correction_camera_settle_seconds", 0.0)) < 0.0:
                raise ValueError(
                    "navigation.bin.side_docking.experimental.post_correction_camera_settle_seconds must be non-negative"
                )
        strategy = self.get("avoidance.strategy", "disabled")
        if strategy not in ("disabled", "scripted", "tangentbug_depth"):
            raise ValueError("unsupported avoidance.strategy: {}".format(strategy))
        max_pickups = int(self.get("runtime.max_pickups", 1))
        if max_pickups < 0:
            raise ValueError("runtime.max_pickups must be zero or positive")
        if bool(self.get("vague_map.enabled", False)):
            vague_map = _require(self.data, "vague_map")
            for name in (
                "bounds_m",
                "initial_pose",
                "bin_marker_position",
                "bin_docking_pose",
                "odometry",
                "navigation",
                "camera_horizontal_fov_rad",
                "depth_to_distance_scale_m_per_unit",
            ):
                if name not in vague_map:
                    raise ValueError("missing config key: vague_map.{}".format(name))
            bounds = vague_map["bounds_m"]
            if float(bounds["min_x"]) >= float(bounds["max_x"]) or float(bounds["min_y"]) >= float(bounds["max_y"]):
                raise ValueError("vague_map.bounds_m must define a positive area")
            odometry = vague_map["odometry"]
            for key in ("linear_meters_per_speed_second", "linear_slip_factor", "angular_radians_per_speed_second"):
                if float(odometry[key]) <= 0.0:
                    raise ValueError("vague_map.odometry.{} must be positive".format(key))
            sample_speeds = set()
            for sample in odometry.get("linear_response_samples", []):
                values = {
                    key: float(_require(sample, key))
                    for key in ("speed", "seconds", "distance_m")
                }
                if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
                    raise ValueError("vague_map.odometry.linear_response_samples must be positive and finite")
                if values["speed"] in sample_speeds:
                    raise ValueError("vague_map.odometry.linear_response_samples speeds must be unique")
                sample_speeds.add(values["speed"])
            navigation = vague_map["navigation"]
            for key in ("turn_speed", "turn_pulse_seconds", "forward_speed", "forward_pulse_seconds", "timeout_seconds"):
                if float(navigation[key]) <= 0.0:
                    raise ValueError("vague_map.navigation.{} must be positive".format(key))
            if int(navigation.get("max_steps", 0)) <= 0:
                raise ValueError("vague_map.navigation.max_steps must be positive")
            localization = vague_map.get("bin_tag_localization", {})
            if bool(localization.get("enabled", False)):
                calibration_yaml = str(_require(self.data, "camera.calibration_yaml")).strip()
                if not calibration_yaml:
                    raise ValueError("camera.calibration_yaml is required for bin tag localization")
                _require(vague_map, "bin_marker_position.heading_rad")
                for key in (
                    "minimum_tag_distance_m", "maximum_tag_distance_m",
                    "maximum_position_error_m", "maximum_heading_error_rad",
                ):
                    value = float(_require(localization, key))
                    if not math.isfinite(value) or value <= 0.0:
                        raise ValueError("vague_map.bin_tag_localization.{} must be positive".format(key))
                if float(localization["minimum_tag_distance_m"]) >= float(localization["maximum_tag_distance_m"]):
                    raise ValueError("bin tag localization distance range must be positive")
                for camera_name in ("front", "side"):
                    mount = _require(localization, "camera_mounts.{}".format(camera_name))
                    for key in ("x_m", "y_m", "heading_rad"):
                        if not math.isfinite(float(_require(mount, key))):
                            raise ValueError("bin tag localization {}.{} must be finite".format(camera_name, key))
        elif max_pickups <= 0:
            raise ValueError("runtime.max_pickups must be positive when vague_map is disabled")
        return True

    def section(self, name):
        return self.data[name]

    def target_navigation(self, target_type):
        name = target_type.value if hasattr(target_type, "value") else str(target_type)
        return self.data["navigation"][name]

    def get(self, path, default=None):
        value = self.data
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    def resolve_path(self, path):
        if not path or os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.project_root, path))


def load_empirical_parameters(path=None, config_path=None):
    _, _, paths, project_root = _runtime_paths(config_path)
    value = Path(path or paths.get("empirical_parameters", "empirical_parameters.json"))
    parameters_path = value if value.is_absolute() else (project_root / value).resolve()
    return _read_json(parameters_path)


def save_empirical_parameters(data, path=None, config_path=None):
    _, _, paths, project_root = _runtime_paths(config_path)
    value = Path(path or paths.get("empirical_parameters", "empirical_parameters.json"))
    parameters_path = value if value.is_absolute() else (project_root / value).resolve()
    with open(str(parameters_path), "w") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    print("[parameters] saved {}".format(parameters_path))
    return str(parameters_path)


def load_config(parameters_path=None, config_path=None, overrides=None):
    config_path, runtime_config, paths, project_root = _runtime_paths(config_path)
    value = Path(parameters_path or paths.get("empirical_parameters", "empirical_parameters.json"))
    parameters_path = value if value.is_absolute() else (project_root / value).resolve()
    data = _deep_merge(_read_json(parameters_path), runtime_config)
    if overrides:
        data = _deep_merge(data, overrides)
    data = _resolve_base_speed_references(data)
    routine_value = Path(paths.get("predefined_routines", "predefined_routines.json"))
    routine_path = routine_value if routine_value.is_absolute() else (project_root / routine_value).resolve()
    data["predefined_routines"] = _read_json(routine_path)
    return DemoConfig(data, config_path, parameters_path, project_root)


_, PROJECT_CONFIG, PROJECT_PATHS, BASE_DIR = _runtime_paths()


def project_path(name, default):
    value = Path(PROJECT_PATHS.get(name, default))
    return value if value.is_absolute() else (BASE_DIR / value).resolve()


EMPIRICAL_PARAMETERS_PATH = project_path("empirical_parameters", "empirical_parameters.json")
PREDEFINED_ROUTINES_PATH = project_path("predefined_routines", "predefined_routines.json")
ASSETS_DIR = project_path("assets", "assets")
TESTS_DIR = project_path("tests", "tests")
CAN_MODEL_DIR = project_path("can_model_directory", "assets/models/detectnet_native_can")
APRILTAG_IMAGE_PATH = project_path("apriltag_image", "assets/bin_apriltag_36h11_id_0.png")
LOG_DIR = project_path("logs", "logs")
DIAGNOSTIC_OUTPUT_DIR = project_path("diagnostic_outputs", "diagnostic_outputs")
