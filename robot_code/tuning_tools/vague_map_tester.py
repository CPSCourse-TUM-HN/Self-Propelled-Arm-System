from __future__ import print_function

import json
import math
import os
import threading
import time

from demo_core.config import load_config
from demo_core.fsm_types import MissionContext, MissionEvent, MissionState
from demo_core.perception import AprilTagBinDetector, DepthSensor, empty_detection
from demo_core.robot_control import BaseController
from demo_core.state_machine import DemoStateMachine, RobotComponents
from demo_core.turn_response import TurnResponseModel
from demo_core.vague_map import Point2D, Pose2D, normalize_heading


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PARAMETERS = os.path.join(PROJECT_ROOT, "tuning_tools", "vague_map_test_parameters.json")


class NoOpArm(object):
    def cancel_motion(self):
        return None

    def pose(self, _name):
        return None


class AlwaysMissingBinDetector(object):
    def load(self):
        return None

    def detect(self, _frame):
        return empty_detection("bin")


class AlwaysMissingCanDetector(object):
    def load(self):
        return None

    def detect(self, _frame):
        return empty_detection("can")

    def detect_all(self, _frame):
        return []


def _finite_number(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{} must be finite".format(name))
    return value


def load_test_parameters(path=DEFAULT_PARAMETERS):
    with open(os.path.abspath(path), "r") as stream:
        data = json.load(stream)
    for section in ("initial_pose", "bin_marker_position", "bin_docking_pose", "odometry", "navigation"):
        if not isinstance(data.get(section), dict):
            raise ValueError("missing object: {}".format(section))
    for name in ("initial_pose", "bin_marker_position", "bin_docking_pose"):
        for key in ("x_m", "y_m"):
            _finite_number(data[name].get(key), "{}.{}".format(name, key))
    _finite_number(data["initial_pose"].get("heading_rad", 0.0), "initial_pose.heading_rad")
    _finite_number(data["bin_docking_pose"].get("heading_rad", 0.0), "bin_docking_pose.heading_rad")
    for key in ("linear_meters_per_speed_second", "linear_slip_factor"):
        if _finite_number(data["odometry"].get(key), "odometry.{}".format(key)) <= 0.0:
            raise ValueError("odometry.{} must be positive".format(key))
    for key in ("turn_speed", "forward_speed", "heading_tolerance_rad", "arrival_tolerance_m", "timeout_seconds"):
        if _finite_number(data["navigation"].get(key), "navigation.{}".format(key)) <= 0.0:
            raise ValueError("navigation.{} must be positive".format(key))
    for key in ("turn_speed", "forward_speed"):
        if float(data["navigation"][key]) > 1.0:
            raise ValueError("navigation.{} must not exceed 1.0".format(key))
    if int(data["navigation"].get("max_steps", 0)) <= 0:
        raise ValueError("navigation.max_steps must be positive")
    calibration = data.get("forward_factor_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("missing object: forward_factor_calibration")
    for key in ("speed", "seconds", "measured_distance_m"):
        if _finite_number(calibration.get(key), "forward_factor_calibration.{}".format(key)) <= 0.0:
            raise ValueError("forward_factor_calibration.{} must be positive".format(key))
    if float(calibration["speed"]) > 1.0:
        raise ValueError("forward_factor_calibration.speed must not exceed 1.0")
    expected = str(data.get("expected_first_motion", "")).lower()
    if expected not in ("left", "right", "forward", "arrived"):
        raise ValueError("expected_first_motion must be left, right, forward, or arrived")
    return data


def first_motion(initial_pose, destination, heading_tolerance_rad, arrival_tolerance_m):
    pose = Pose2D.from_mapping(initial_pose)
    point = Point2D.from_mapping(destination)
    distance = pose.distance_to(point)
    if distance <= float(arrival_tolerance_m):
        return {"motion": "arrived", "distance_m": distance, "heading_error_rad": 0.0}
    desired_heading = math.atan2(point.x_m - pose.x_m, point.y_m - pose.y_m)
    error = normalize_heading(desired_heading - pose.heading_rad)
    if abs(error) <= float(heading_tolerance_rad):
        motion = "forward"
    else:
        motion = "right" if error > 0.0 else "left"
    return {"motion": motion, "distance_m": distance, "heading_error_rad": error}


def calculate_forward_factor(measured_distance_m, requested_speed, seconds, command_scale=1.0, slip_factor=1.0):
    measured_distance_m = _finite_number(measured_distance_m, "measured_distance_m")
    effective_speed = _finite_number(requested_speed, "requested_speed") * _finite_number(command_scale, "command_scale")
    seconds = _finite_number(seconds, "seconds")
    slip_factor = _finite_number(slip_factor, "slip_factor")
    if measured_distance_m <= 0.0 or effective_speed <= 0.0 or seconds <= 0.0 or slip_factor <= 0.0:
        raise ValueError("distance, effective speed, seconds, and slip factor must be positive")
    return measured_distance_m / (effective_speed * seconds * slip_factor)


def build_test_config(data, real_base=False, real_camera=False):
    navigation = dict(data["navigation"])
    loop_pause = float(navigation.pop("loop_pause_seconds", 0.05))
    turn_speed = float(navigation.pop("turn_speed"))
    forward_speed = float(navigation.pop("forward_speed"))
    navigation["turn_speed"] = "turn.slow"
    navigation["forward_speed"] = "linear.fast"
    overrides = {
        "runtime": {
            "dry_run": {"base": not bool(real_base), "camera": not bool(real_camera), "arm": True},
            "loop_pause_seconds": loop_pause,
        },
        "base_motion_speed_profiles": {
            "turn": {"slow": turn_speed},
            "linear": {"fast": forward_speed},
        },
        "vague_map": {
            "enabled": True,
            "initial_pose": dict(data["initial_pose"]),
            "bin_marker_position": dict(data["bin_marker_position"]),
            "bin_docking_pose": dict(data["bin_docking_pose"]),
            "odometry": dict(data["odometry"], linear_response_samples=[]),
            "navigation": navigation,
        },
    }
    return load_config(overrides=overrides)


def calibration_summary(data, config):
    calibration = data.get("forward_factor_calibration", {})
    scale = float(config.get("base_motion_command_scales.forward", 1.0))
    factor = calculate_forward_factor(
        calibration.get("measured_distance_m"),
        calibration.get("speed"),
        calibration.get("seconds"),
        scale,
        data["odometry"].get("linear_slip_factor", 1.0),
    )
    return {
        "requested_speed": float(calibration["speed"]),
        "command_scale": scale,
        "effective_speed": float(calibration["speed"]) * scale,
        "seconds": float(calibration["seconds"]),
        "measured_distance_m": float(calibration["measured_distance_m"]),
        "linear_meters_per_speed_second": factor,
    }


def preview(data, config=None):
    config = config or build_test_config(data)
    result = first_motion(
        data["initial_pose"], data["bin_docking_pose"],
        data["navigation"]["heading_tolerance_rad"],
        data["navigation"]["arrival_tolerance_m"],
    )
    expected = str(data["expected_first_motion"]).lower()
    result["expected_first_motion"] = expected
    result["matches_expectation"] = result["motion"] == expected
    turn_model = TurnResponseModel(
        config.section("base_turn_response"),
        config.get("vague_map.odometry.angular_radians_per_speed_second", math.pi),
    )
    if result["motion"] in ("left", "right"):
        result["estimated_turn_seconds"] = turn_model.seconds_for_angle(
            abs(result["heading_error_rad"]), result["motion"], data["navigation"]["turn_speed"]
        )
    else:
        result["estimated_turn_seconds"] = 0.0
    return result


class VagueMapTestSession(object):
    def __init__(self, parameters_path=DEFAULT_PARAMETERS):
        self.parameters_path = os.path.abspath(parameters_path)
        self.data = None
        self.config = None
        self.runtime = None
        self.stop_event = threading.Event()

    def reload(self, real_base=False, real_camera=False):
        self.stop()
        self.data = load_test_parameters(self.parameters_path)
        self.config = build_test_config(self.data, real_base=real_base, real_camera=real_camera)
        return {"preview": preview(self.data, self.config), "forward_calibration": calibration_summary(self.data, self.config)}

    def _make_runtime(self, force_tag_missing):
        base = BaseController(self.config)
        depth = DepthSensor(self.config)
        bin_detector = AlwaysMissingBinDetector() if force_tag_missing else AprilTagBinDetector(self.config)
        services = RobotComponents(base, NoOpArm(), depth, AlwaysMissingCanDetector(), bin_detector)
        context = MissionContext()
        context.grabbed = True
        runtime = DemoStateMachine(self.config, services=services, context=context)
        runtime.state = MissionState.PLANNING
        context.begin_state(MissionState.PLANNING)
        return runtime

    def run_direct_bin(self, force_tag_missing=True):
        if self.config is None:
            self.reload()
        prediction = preview(self.data, self.config)
        if not prediction["matches_expectation"]:
            raise RuntimeError("predicted first motion {} does not match expected {}; edit JSON or expectation".format(
                prediction["motion"], prediction["expected_first_motion"]
            ))
        self.stop_event.clear()
        self.runtime = self._make_runtime(force_tag_missing)
        runtime = self.runtime
        runtime.services.depth.start(camera_only=True)
        runtime.services.bin_detector.load()
        started = time.time()
        records = []
        try:
            planning = runtime.step_once()
            if planning.event != MissionEvent.MAP_TARGET_AVAILABLE or runtime.state != MissionState.MAP_NAVIGATING:
                raise RuntimeError("Planning did not override search with MAP_NAVIGATING")
            while not self.stop_event.is_set():
                if time.time() - started >= float(self.data["navigation"]["timeout_seconds"]):
                    records.append({"result": "tester_timeout", "pose": runtime.vague_map.robot_pose.as_dict()})
                    break
                outcome = runtime.step_once()
                snapshot = runtime.services.base.command_snapshot()
                records.append({
                    "state": runtime.state.value,
                    "event": outcome.event.value if outcome.event is not None else None,
                    "command": snapshot,
                    "pose": runtime.vague_map.robot_pose.as_dict(),
                })
                if outcome.event == MissionEvent.TARGET_FOUND:
                    records[-1]["result"] = "tag_found_visual_takeover"
                    break
                if outcome.event == MissionEvent.MAP_DESTINATION_REACHED:
                    records[-1]["result"] = "map_destination_reached_before_search"
                    break
                if runtime.state not in (MissionState.MAP_NAVIGATING,):
                    records[-1]["result"] = "unexpected_state_exit"
                    break
                time.sleep(float(self.data["navigation"].get("loop_pause_seconds", 0.05)))
        finally:
            runtime.services.base.stop()
            runtime.services.depth.stop()
        return records

    def run_forward_calibration_pulse(self):
        if self.config is None:
            self.reload()
        calibration = self.data["forward_factor_calibration"]
        base = BaseController(self.config)
        try:
            base.pulse("forward", float(calibration["speed"]), float(calibration["seconds"]), "vague_map_forward_factor")
        finally:
            base.stop()

    def stop(self):
        self.stop_event.set()
        if self.runtime is not None:
            self.runtime.services.base.stop()
            self.runtime.services.depth.stop()
            self.runtime = None


def build_ui(parameters_path=DEFAULT_PARAMETERS):
    import ipywidgets as widgets
    from IPython.display import display

    session = VagueMapTestSession(parameters_path)
    output = widgets.Output(layout=widgets.Layout(border="1px solid #b8b8b8", height="360px", overflow_y="auto"))
    real_base = widgets.Checkbox(value=False, description="Real base motion")
    real_camera = widgets.Checkbox(value=False, description="Real camera")
    force_missing = widgets.Checkbox(value=True, description="Force Tag missing")
    reload_button = widgets.Button(description="Reload JSON", button_style="info")
    preview_button = widgets.Button(description="Preview / Assert")
    forward_button = widgets.Button(description="Forward factor pulse", button_style="warning")
    run_button = widgets.Button(description="Run direct bin", button_style="success")
    stop_button = widgets.Button(description="STOP BASE", button_style="danger")
    state = {"thread": None}

    def write(value):
        output.append_stdout(json.dumps(value, indent=2, sort_keys=True) + "\n")

    def reload_params(_=None):
        try:
            write(session.reload(real_base.value, real_camera.value))
        except Exception as exc:
            output.append_stderr("[error] {}\n".format(exc))

    def show_preview(_=None):
        try:
            if session.data is None:
                session.reload(real_base.value, real_camera.value)
            write({"preview": preview(session.data, session.config), "forward_calibration": calibration_summary(session.data, session.config)})
        except Exception as exc:
            output.append_stderr("[error] {}\n".format(exc))

    def worker(mode):
        try:
            session.reload(real_base.value, real_camera.value)
            if mode == "forward":
                session.run_forward_calibration_pulse()
                write({"result": "pulse_complete", "next": "measure distance, edit measured_distance_m, then Reload JSON"})
            else:
                records = session.run_direct_bin(force_tag_missing=force_missing.value)
                write({"result": records[-1] if records else "stopped", "records": len(records)})
        except Exception as exc:
            output.append_stderr("[error] {}\n".format(exc))
        finally:
            state["thread"] = None

    def start(mode):
        if state["thread"] is not None and state["thread"].is_alive():
            output.append_stdout("[run] already running\n")
            return
        thread = threading.Thread(target=worker, args=(mode,))
        thread.daemon = True
        state["thread"] = thread
        thread.start()

    reload_button.on_click(reload_params)
    preview_button.on_click(show_preview)
    forward_button.on_click(lambda _: start("forward"))
    run_button.on_click(lambda _: start("navigate"))
    stop_button.on_click(lambda _: session.stop())
    panel = widgets.VBox([
        widgets.HTML("<b>Vague-map direct-bin tester</b>"),
        widgets.HTML("Edit <code>tuning_tools/vague_map_test_parameters.json</code>, reload, preview the expected first motion, then explicitly enable real hardware."),
        widgets.HBox([real_base, real_camera, force_missing]),
        widgets.HBox([reload_button, preview_button, forward_button, run_button, stop_button]),
        output,
    ])
    reload_params()
    display(panel)
    return panel
