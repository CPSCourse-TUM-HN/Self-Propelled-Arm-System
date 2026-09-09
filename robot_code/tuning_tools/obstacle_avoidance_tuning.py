from __future__ import print_function

import csv
import json
import os
import threading
import time
import traceback

import cv2
import numpy as np

from demo_core.turn_response import TurnResponseModel


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_PARAMETERS = os.path.join(SCRIPT_DIR, "obstacle_avoidance_parameters.json")


def load_avoidance_parameters(path=DEFAULT_PARAMETERS):
    with open(path, "r") as stream:
        data = json.load(stream)
    required = ("camera", "perception", "motion", "control", "safety", "media")
    for key in required:
        if not isinstance(data.get(key), dict):
            raise ValueError("missing parameter section: {}".format(key))
    top_ignore_ratio = float(data["perception"].get("obstacle_top_ignore_ratio", 0.0))
    if top_ignore_ratio < 0.0 or top_ignore_ratio >= 0.9:
        raise ValueError("perception.obstacle_top_ignore_ratio must be in [0.0, 0.9)")
    bottom_ignore_ratio = float(data["perception"].get("obstacle_bottom_ignore_ratio", 0.0))
    if bottom_ignore_ratio < 0.0 or bottom_ignore_ratio >= 0.9:
        raise ValueError("perception.obstacle_bottom_ignore_ratio must be in [0.0, 0.9)")
    if top_ignore_ratio + bottom_ignore_ratio >= 0.95:
        raise ValueError("obstacle top and bottom ignore ratios leave too little detection area")
    motion = data["motion"]
    for name in (
        "linear_speed", "turn_speed", "linear_meters_per_speed_second",
        "angular_radians_per_speed_second",
    ):
        if float(motion[name]) <= 0.0:
            raise ValueError("motion.{} must be positive".format(name))
    if float(motion.get("depth_to_segment_time_factor", 0.0)) < 0.0:
        raise ValueError("motion.depth_to_segment_time_factor must be non-negative")
    if float(motion.get("pass_depth_to_time_factor", 0.0)) < 0.0:
        raise ValueError("motion.pass_depth_to_time_factor must be non-negative")
    if motion.get("side_override", "auto") not in ("auto", "left", "right"):
        raise ValueError("side_override must be auto, left, or right")
    if float(motion["maximum_offset_seconds"]) < float(motion["minimum_offset_seconds"]):
        raise ValueError("maximum_offset_seconds must be >= minimum_offset_seconds")
    if float(motion["pass_max_seconds"]) < float(motion["pass_min_seconds"]):
        raise ValueError("pass_max_seconds must be >= pass_min_seconds")
    if float(data["control"]["control_fps"]) <= 0.0:
        raise ValueError("control_fps must be positive")
    for name in ("obstacle_confirmation_frames", "linear_obstacle_confirmation_frames"):
        if int(data["control"].get(name, 3)) <= 0:
            raise ValueError("{} must be positive".format(name))
    for name in (
        "blocked_failure_frames", "post_turn_clear_frames", "tangent_safe_frames",
        "maximum_lateral_corrections",
    ):
        if int(data["control"].get(name, 1)) <= 0:
            raise ValueError("control.{} must be positive".format(name))
    percentile = float(data["perception"].get("center_depth_percentile", 20.0))
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError("center_depth_percentile must be between 0 and 100")
    wall_width = float(data["perception"].get("wall_width_threshold_norm", 0.6))
    if wall_width <= 0.0 or wall_width > 1.0:
        raise ValueError("perception.wall_width_threshold_norm must be in (0.0, 1.0]")
    tracking = data["perception"]
    max_shift = float(tracking.get("tracking_max_center_shift_norm", 0.2))
    min_area = float(tracking.get("tracking_min_area_ratio", 0.45))
    max_area = float(tracking.get("tracking_max_area_ratio", 2.2))
    depth_tolerance = float(tracking.get("tracking_depth_tolerance_ratio", 0.45))
    if max_shift <= 0.0 or max_shift > 1.5:
        raise ValueError("perception.tracking_max_center_shift_norm must be in (0.0, 1.5]")
    if min_area <= 0.0 or max_area < min_area:
        raise ValueError("perception tracking area ratios are invalid")
    if depth_tolerance < 0.0:
        raise ValueError("perception.tracking_depth_tolerance_ratio must be non-negative")
    return data


class ObstacleAvoidanceSession(object):
    STATES = (
        "READY", "CRUISING", "DETECTED", "TURN_TO_TANGENT", "OFFSET_OUT",
        "TURN_PARALLEL", "PASSING", "TURN_TO_LINE", "OFFSET_RETURN",
        "RESTORE_HEADING", "VERIFY_CLEAR", "DONE", "FAILED", "STOPPED",
    )

    def __init__(self, parameters_path=DEFAULT_PARAMETERS, base_real=False, camera_real=False):
        from demo_core import load_config
        from demo_core.perception import DepthSensor
        from demo_core.robot_control import BaseController
        from demo_core.tangentbug import DepthTangentBugPlanner

        self.parameters_path = os.path.abspath(parameters_path)
        self.parameters = load_avoidance_parameters(self.parameters_path)
        self.config = load_config(overrides={
            "runtime": {"dry_run": {"base": not bool(base_real), "camera": not bool(camera_real)}},
            "camera": {
                "width": int(self.parameters["camera"]["width"]),
                "height": int(self.parameters["camera"]["height"]),
                "depth_network": str(self.parameters["camera"]["depth_network"]),
                "depth_enabled": True,
            },
        })
        self.base = BaseController(self.config)
        self.turn_response = TurnResponseModel(
            self.config.section("base_turn_response"),
            self.parameters["motion"]["angular_radians_per_speed_second"],
        )
        self.depth = DepthSensor(self.config)
        self.planner = DepthTangentBugPlanner(self.parameters["perception"])
        self.state = "READY"
        self.selected_side = None
        self.tangent_turn_seconds = None
        self.offset_out_seconds = None
        self.offset_segments = []
        self.return_segments = None
        self.active_return_segment = None
        self.lateral_correction_count = 0
        self.initial_obstacle_depth = None
        self.locked_obstacle = None
        self.depth_scaled_segment_seconds = 0.0
        self.pass_depth_scaled_seconds = 0.0
        self.motion_interrupted_by_obstacle = None
        self.estimated_heading_rad = 0.0
        self.estimated_lateral_offset_m = 0.0
        self.stop_event = threading.Event()
        self.worker = None
        self.worker_name = None
        self.worker_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.camera_started = False
        self.live_enabled = True
        self.record_enabled = False
        self.geometry_hud_enabled = True
        self.text_hud_enabled = True
        self.frame_callback = None
        self.log_callback = None
        self.writer = None
        self.recording_path = None
        self.last_analysis = None
        self.last_frame = None
        self.last_rendered = None
        self.last_preview_at = 0.0
        self.last_record_at = 0.0
        self.session_started_at = time.time()
        self._open_logs()

    def _open_logs(self):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        root = self.config.resolve_path(self.config.get("paths.logs", "logs"))
        self.output_directory = os.path.join(root, "obstacle_avoidance_tuning", stamp)
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory)
        self.log_path = os.path.join(self.output_directory, "session.log")
        self.csv_path = os.path.join(self.output_directory, "frames.csv")
        self.csv_stream = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.DictWriter(self.csv_stream, fieldnames=(
            "timestamp", "state", "status", "detected", "path_blocked", "unsafe", "selected_side",
            "background_depth", "obstacle_depth", "width_norm", "height_norm",
            "center_depth", "center_x_norm", "bbox_left_norm", "bbox_right_norm", "reason",
        ))
        self.csv_writer.writeheader()
        self.csv_stream.flush()

    def log(self, message):
        line = "{} {}".format(time.strftime("%H:%M:%S"), message)
        with open(self.log_path, "a") as stream:
            stream.write(line + "\n")
        if self.log_callback is not None:
            self.log_callback(line)
        else:
            print(line)

    def reload_parameters(self):
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError("stop the active job before reloading parameters")
        from demo_core.tangentbug import DepthTangentBugPlanner
        self.parameters = load_avoidance_parameters(self.parameters_path)
        self.planner = DepthTangentBugPlanner(self.parameters["perception"])
        self.log("[parameters] reloaded {}".format(self.parameters_path))
        return self.parameters

    def configure_media(self, live=True, record=False, geometry_hud=True, text_hud=True):
        self.live_enabled = bool(live)
        self.record_enabled = bool(record)
        self.geometry_hud_enabled = bool(geometry_hud)
        self.text_hud_enabled = bool(text_hud)

    def start_camera(self):
        if not self.camera_started:
            self.depth.start(camera_only=False)
            self.camera_started = True
            self.log(
                "[camera] DepthNet camera started frame={}x{} obstacle_roi_ignore=(top={:.3f}, bottom={:.3f})".format(
                    int(self.parameters["camera"]["width"]),
                    int(self.parameters["camera"]["height"]),
                    float(self.parameters["perception"].get("obstacle_top_ignore_ratio", 0.0)),
                    float(self.parameters["perception"].get("obstacle_bottom_ignore_ratio", 0.0)),
                )
            )
        return True

    def stop_base(self):
        self.stop_event.set()
        self.base.stop()
        self.log("[stop] base stop requested")

    def stop_media(self):
        self.stop_event.set()
        self.base.stop()
        worker_alive = self.worker is not None and self.worker.is_alive()
        if not worker_alive:
            self._close_writer()
        self.log("[media] stop requested cleanup_owner={}".format(
            "worker" if worker_alive else "caller"
        ))
        return {"stopping": worker_alive, "recording_path": self.recording_path}

    def release_camera(self):
        self.stop_event.set()
        self.base.stop()
        worker = self.worker
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(5.0)
        if worker is not None and worker.is_alive():
            self.log("[camera] worker still stopping; camera release deferred")
            return {"released": False, "reason": "worker still stopping"}
        self._close_writer()
        self.depth.stop()
        self.camera_started = False
        self.log("[camera] stopped and released")
        return {"released": True}

    def close(self):
        result = self.release_camera()
        if not result.get("released", False):
            raise RuntimeError("session worker is still stopping; reset is deferred")
        if self.csv_stream is not None:
            self.csv_stream.close()
            self.csv_stream = None

    def _ensure_writer(self):
        if not self.record_enabled or self.writer is not None:
            return
        settings = self.parameters["media"]
        self.recording_path = os.path.join(
            self.output_directory,
            "obstacle_avoidance_{}.avi".format(int(time.time() * 1000)),
        )
        codec = str(settings.get("video_codec", "MJPG"))
        size = (int(settings["record_width"]), int(settings["record_height"]))
        self.writer = cv2.VideoWriter(
            self.recording_path,
            cv2.VideoWriter_fourcc(*codec),
            float(settings["record_fps"]),
            size,
        )
        if not self.writer.isOpened():
            self.writer.release()
            self.writer = None
            raise RuntimeError("could not open recording {}".format(self.recording_path))
        self.log("[media] recording started {}".format(self.recording_path))

    def _close_writer(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.log("[media] recording finalized {}".format(self.recording_path))

    def _write_frame_row(self, analysis):
        row = analysis.as_dict()
        row.update({"timestamp": time.time(), "state": self.state})
        self.csv_writer.writerow({key: row.get(key) for key in self.csv_writer.fieldnames})
        self.csv_stream.flush()

    def _render(self, frame, analysis):
        rendered = self.planner.draw_obstacle_debug(
            frame,
            analysis,
            self.state,
            self.record_enabled,
            draw_geometry=self.geometry_hud_enabled,
            draw_text=self.text_hud_enabled,
        )
        settings = self.parameters["media"]
        now = time.time()
        self._ensure_writer()
        if self.writer is not None:
            interval = 1.0 / max(0.1, float(settings["record_fps"]))
            if now - self.last_record_at >= interval:
                record_frame = cv2.resize(
                    rendered,
                    (int(settings["record_width"]), int(settings["record_height"])),
                    interpolation=cv2.INTER_LINEAR,
                )
                self.writer.write(record_frame)
                self.last_record_at = now
        if self.live_enabled and self.frame_callback is not None:
            interval = 1.0 / max(0.1, float(settings["preview_fps"]))
            if now - self.last_preview_at >= interval:
                preview = cv2.resize(
                    rendered,
                    (int(settings["preview_width"]), int(settings["preview_height"])),
                    interpolation=cv2.INTER_AREA,
                )
                ok, encoded = cv2.imencode(
                    ".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), int(settings["jpeg_quality"])]
                )
                if ok:
                    self.frame_callback(encoded.tobytes())
                self.last_preview_at = now
        self.last_rendered = rendered

    def sample(self):
        self.start_camera()
        with self.inference_lock:
            frame = self.depth.read_frame()
            if frame is None:
                raise RuntimeError("camera frame unavailable")
            frame = self.depth.depth_input_frame(frame)
            depth_map = self.depth.depth_map_frame(frame, frame_space=self.depth.depth_frame_space)
            analysis = self.planner.analyze_obstacle(depth_map)
        self.last_frame = frame
        self.last_analysis = analysis
        self._write_frame_row(analysis)
        self._render(frame, analysis)
        return analysis

    def analyze_frame(self):
        analysis = self.sample()
        self.log("[analysis] {}".format(analysis.as_dict()))
        return analysis.as_dict()

    def _emergency_close(self, analysis):
        if getattr(analysis, "status", None) == "IGNORED_WALL":
            return False
        if analysis.obstacle_depth is None:
            return False
        return float(analysis.obstacle_depth) <= float(self.parameters["safety"]["emergency_depth"])

    @staticmethod
    def _absolute_depth_blocked(analysis):
        return (
            getattr(analysis, "status", None) == "BLOCKED"
            and getattr(analysis, "reason", None) in (
                "absolute center depth stop",
                "wide wall reached absolute center stop",
            )
        )

    def _sleep_control(self):
        self.stop_event.wait(1.0 / max(0.2, float(self.parameters["control"]["control_fps"])))

    def _run_visual_motion(
        self, direction, speed, timeout, label, stop_condition,
        minimum_seconds=0.0, ignore_obstacle_failures=False,
        confirm_new_obstacles=False,
    ):
        started = time.time()
        self.motion_interrupted_by_obstacle = None
        obstacle_missing = 0
        frame_missing = 0
        frame_missing_limit = max(0, int(self.parameters["control"]["missing_frame_limit"]))
        self.base.start_motion(direction, speed, label)
        try:
            while time.time() - started < float(timeout):
                if self.stop_event.is_set():
                    raise RuntimeError("stop requested")
                try:
                    analysis = self.sample()
                    frame_missing = 0
                except RuntimeError as exc:
                    if "camera frame unavailable" not in str(exc):
                        raise
                    frame_missing += 1
                    self.log("[camera] missing frame {}/{}".format(frame_missing, frame_missing_limit))
                    if frame_missing > frame_missing_limit:
                        raise RuntimeError("camera missing-frame limit exceeded")
                    self._sleep_control()
                    continue
                if ignore_obstacle_failures and self._absolute_depth_blocked(analysis):
                    raise RuntimeError("absolute depth protection reached: {}".format(analysis.reason))
                if not ignore_obstacle_failures and self._emergency_close(analysis):
                    raise RuntimeError("emergency depth reached")
                if getattr(analysis, "status", None) == "IGNORED_WALL":
                    self.log("[wall] frame treated as clear path")
                if getattr(analysis, "status", None) == "CANDIDATE":
                    if label == "avoidance_cruise" or confirm_new_obstacles:
                        confirmation_frames = None
                        if confirm_new_obstacles:
                            confirmation_frames = self.parameters["control"].get(
                                "linear_obstacle_confirmation_frames", 3
                            )
                        paused_at = time.time()
                        analysis = self._confirm_candidate_while_stopped(
                            analysis,
                            direction,
                            speed,
                            label,
                            confirmation_frames=confirmation_frames,
                        )
                        started += time.time() - paused_at
                        if (
                            confirm_new_obstacles
                            and analysis.path_blocked
                            and getattr(analysis, "status", None) == "OBSTACLE"
                        ):
                            elapsed = time.time() - started
                            self.motion_interrupted_by_obstacle = analysis
                            self.log(
                                "[interrupt] {} confirmed new obstacle after "
                                "moving_seconds={:.3f}".format(label, elapsed)
                            )
                            return elapsed
                    else:
                        analysis.status = "OBSTACLE"
                        analysis.reason = "confirmed bbox tracked during bypass"
                if (
                    not ignore_obstacle_failures
                    and (analysis.unsafe or getattr(analysis, "status", None) == "BLOCKED")
                ):
                    raise RuntimeError("blocked obstacle geometry: {}".format(analysis.reason))
                if analysis.detected:
                    obstacle_missing = 0
                else:
                    obstacle_missing += 1
                elapsed = time.time() - started
                if elapsed >= float(minimum_seconds) and stop_condition(analysis, obstacle_missing, elapsed):
                    return elapsed
                self._sleep_control()
            raise RuntimeError("{} timeout".format(label))
        finally:
            self.base.stop()

    @staticmethod
    def _bbox_center_and_size(bbox):
        x, y, width, height = [float(value) for value in bbox]
        return x + width / 2.0, y + height / 2.0, width, height

    def _bbox_matches_reference(self, candidate_bbox, reference_bbox, image_shape):
        if candidate_bbox is None or reference_bbox is None or image_shape is None:
            return False
        image_height, image_width = image_shape[:2]
        old_x, old_y, old_width, old_height = self._bbox_center_and_size(reference_bbox)
        new_x, new_y, new_width, new_height = self._bbox_center_and_size(candidate_bbox)
        center_shift = (
            ((new_x - old_x) / float(max(1, image_width))) ** 2
            + ((new_y - old_y) / float(max(1, image_height))) ** 2
        ) ** 0.5
        settings = self.parameters["perception"]
        if center_shift > float(settings.get("tracking_max_center_shift_norm", 0.2)):
            return False
        old_area = max(1.0, old_width * old_height)
        area_ratio = max(1.0, new_width * new_height) / old_area
        if area_ratio < float(settings.get("tracking_min_area_ratio", 0.45)):
            return False
        if area_ratio > float(settings.get("tracking_max_area_ratio", 2.2)):
            return False
        return True

    def _find_tracked_bbox(self, analysis, reference_bbox, enforce_depth=True):
        if analysis is None or analysis.mask is None or reference_bbox is None:
            return None
        candidates = list(getattr(analysis, "candidate_bboxes", None) or [])
        known = [tuple(item) for item in candidates]
        if analysis.bbox is not None and tuple(analysis.bbox) not in known:
            candidates.append(analysis.bbox)
        matches = [
            tuple(item) for item in candidates
            if self._bbox_matches_reference(item, reference_bbox, analysis.mask.shape)
        ]
        if not matches:
            return None
        old_x, old_y, _old_width, _old_height = self._bbox_center_and_size(reference_bbox)
        matches.sort(key=lambda item: (
            self._bbox_center_and_size(item)[0] - old_x
        ) ** 2 + (
            self._bbox_center_and_size(item)[1] - old_y
        ) ** 2)
        tracked = matches[0]
        selected = tuple(analysis.bbox) if analysis.bbox is not None else None
        locked_depth = None if self.locked_obstacle is None else self.locked_obstacle.get("obstacle_depth")
        current_depth = analysis.obstacle_depth
        if enforce_depth and tracked == selected and locked_depth is not None and current_depth is not None:
            settings = self.parameters["perception"]
            relative_depth_change = abs(float(current_depth) - float(locked_depth)) / max(
                0.001, abs(float(locked_depth))
            )
            if relative_depth_change > float(settings.get("tracking_depth_tolerance_ratio", 0.45)):
                return None
        return tracked

    @staticmethod
    def _median_bbox(observations):
        values = np.asarray([item.bbox for item in observations], dtype=np.float32)
        return tuple(int(round(value)) for value in np.median(values, axis=0))

    def _lock_confirmed_obstacle(self, observations):
        bbox = self._median_bbox(observations)
        depths = [float(item.obstacle_depth) for item in observations if item.obstacle_depth is not None]
        sides = [item.selected_side for item in observations if item.selected_side in ("left", "right")]
        if sides:
            selected_side = "left" if sides.count("left") > sides.count("right") else "right"
        else:
            selected_side = observations[-1].selected_side
        self.locked_obstacle = {
            "bbox": bbox,
            "tracked_bbox": bbox,
            "obstacle_depth": float(np.median(depths)) if depths else None,
            "selected_side": selected_side,
            "confirmation_frames": len(observations),
        }
        resolved = observations[-1]
        resolved.bbox = bbox
        resolved.center = (
            int(round(bbox[0] + bbox[2] / 2.0)),
            int(round(bbox[1] + bbox[3] / 2.0)),
        )
        resolved.obstacle_depth = self.locked_obstacle["obstacle_depth"]
        resolved.selected_side = selected_side
        resolved.status = "OBSTACLE"
        resolved.reason = "same bbox confirmed for {} frames".format(len(observations))
        return resolved

    def _confirm_candidate_while_stopped(
        self, analysis, direction, speed, label, confirmation_frames=None
    ):
        if confirmation_frames is None:
            confirmation_frames = self.parameters["control"].get(
                "obstacle_confirmation_frames", 3
            )
        needed = max(1, int(confirmation_frames))
        self.base.stop()
        self.log("[candidate] {} stopped for bbox confirmation 1/{}".format(label, needed))
        current = analysis
        observations = [analysis]
        tracked_bbox = analysis.bbox
        for index in range(1, needed):
            if self.stop_event.is_set():
                raise RuntimeError("stop requested")
            self._sleep_control()
            current = self.sample()
            if self._emergency_close(current):
                raise RuntimeError("emergency depth reached")
            status = getattr(current, "status", None)
            if current.unsafe or status == "BLOCKED":
                raise RuntimeError("blocked obstacle geometry: {}".format(current.reason))
            if status == "IGNORED_WALL":
                self.log("[candidate] {} reclassified as wall; confirmation cancelled".format(label))
                self.base.start_motion(direction, speed, label + "_resume")
                return current
            if not current.path_blocked:
                self.log("[candidate] {} left center corridor after {} frames".format(label, index + 1))
                self.base.start_motion(direction, speed, label + "_resume")
                return current
            matched_bbox = self._find_tracked_bbox(current, tracked_bbox, enforce_depth=False)
            if matched_bbox is None or tuple(current.bbox) != tuple(matched_bbox):
                self.log("[candidate] {} bbox identity changed after {} frames".format(label, index + 1))
                current.detected = False
                current.path_blocked = False
                current.status = "UNCONFIRMED"
                current.reason = "bbox identity changed during stationary confirmation"
                self.base.start_motion(direction, speed, label + "_resume")
                return current
            current.bbox = matched_bbox
            observations.append(current)
            tracked_bbox = current.bbox
            self.log("[candidate] {} confirmation {}/{}".format(label, index + 1, needed))
        current = self._lock_confirmed_obstacle(observations)
        self.log(
            "[candidate] {} locked frames={} bbox={} depth={} side={}".format(
                label,
                needed,
                self.locked_obstacle["bbox"],
                self.locked_obstacle["obstacle_depth"],
                self.locked_obstacle["selected_side"],
            )
        )
        return current

    def _run_timed_motion(self, direction, speed, seconds, label, require_clear=False):
        def stop_condition(analysis, _missing, elapsed):
            return elapsed >= float(seconds) and (not require_clear or not analysis.path_blocked)

        control = self.parameters["control"]
        frame_period = 1.0 / max(0.2, float(control["control_fps"]))
        grace = max(float(control.get("timed_motion_grace_seconds", 1.0)), 2.0 * frame_period)
        if require_clear:
            grace += max(1, int(control.get("blocked_failure_frames", 5))) * frame_period
        timeout = float(seconds) + grace
        self.log(
            "[motion] {} direction={} speed={:.3f} target_seconds={:.3f} "
            "timeout_seconds={:.3f}".format(
                label, direction, float(speed), float(seconds), timeout
            )
        )
        actual = self._run_visual_motion(
            direction,
            speed,
            timeout,
            label,
            stop_condition,
            minimum_seconds=seconds,
        )
        self.log(
            "[motion] {} complete direction={} target_seconds={:.3f} "
            "actual_seconds={:.3f} delta_seconds={:.3f}".format(
                label,
                direction,
                float(seconds),
                float(actual),
                float(actual) - float(seconds),
            )
        )
        return actual

    def _verify_clear_while_stopped(self, label):
        blocked_needed = max(1, int(self.parameters["control"].get("blocked_failure_frames", 5)))
        clear_needed = max(1, int(self.parameters["control"].get("post_turn_clear_frames", 3)))
        missing_limit = max(0, int(self.parameters["control"]["missing_frame_limit"]))
        valid_frames = 0
        maximum_attempts = 2 * (blocked_needed + clear_needed) + missing_limit
        attempts = 0
        blocked_streak = 0
        clear_streak = 0
        self.base.stop()
        while attempts < maximum_attempts:
            if self.stop_event.is_set():
                raise RuntimeError("stop requested")
            attempts += 1
            analysis = self.sample()
            if analysis.unsafe or self._emergency_close(analysis):
                raise RuntimeError("{} unsafe after motion".format(label))
            valid_frames += 1
            wall_clear = getattr(analysis, "status", None) == "IGNORED_WALL"
            if analysis.path_blocked and not wall_clear:
                blocked_streak += 1
                clear_streak = 0
                self.log(
                    "[check] {} blocked_streak={}/{} clear_streak=0/{}".format(
                        label, blocked_streak, blocked_needed, clear_needed
                    )
                )
                if blocked_streak >= blocked_needed:
                    self.log("[check] {} persistently blocked".format(label))
                    return False
            else:
                blocked_streak = 0
                clear_streak += 1
                source = "wall-as-clear" if wall_clear else "clear"
                self.log(
                    "[check] {} {} clear_streak={}/{}".format(
                        label, source, clear_streak, clear_needed
                    )
                )
                if clear_streak >= clear_needed:
                    self.log(
                        "[check] {} accepted after {} valid samples".format(label, valid_frames)
                    )
                    return True
            self._sleep_control()
        self.log(
            "[check] {} inconclusive after {} valid samples; "
            "clear_streak={}/{} blocked_streak={}/{}".format(
                label,
                valid_frames,
                clear_streak,
                clear_needed,
                blocked_streak,
                blocked_needed,
            )
        )
        return False

    def _activate_locked_obstacle(self, analysis):
        if self.locked_obstacle is None:
            raise RuntimeError("confirmed obstacle was not locked")
        self.initial_obstacle_depth = self.locked_obstacle["obstacle_depth"]
        factor = float(self.parameters["motion"].get("depth_to_segment_time_factor", 0.0))
        self.depth_scaled_segment_seconds = (
            max(0.0, float(self.initial_obstacle_depth) * factor)
            if self.initial_obstacle_depth is not None
            else 0.0
        )
        pass_factor = float(
            self.parameters["motion"].get("pass_depth_to_time_factor", factor)
        )
        self.pass_depth_scaled_seconds = (
            max(0.0, float(self.initial_obstacle_depth) * pass_factor)
            if self.initial_obstacle_depth is not None
            else 0.0
        )
        requested = self.parameters["motion"].get("side_override", "auto")
        self.selected_side = self.locked_obstacle["selected_side"] if requested == "auto" else requested
        self.state = "DETECTED"
        self.log(
            "[phase] DETECTED selected_side={} locked_obstacle_depth={} "
            "depth_time_factor={:.3f} segment_base_seconds={:.3f} "
            "pass_depth_time_factor={:.3f} pass_base_seconds={:.3f} geometry={}".format(
                self.selected_side,
                self.initial_obstacle_depth,
                factor,
                self.depth_scaled_segment_seconds,
                pass_factor,
                self.pass_depth_scaled_seconds,
                analysis.as_dict(),
            )
        )

    def _phase_cruise(self):
        self.state = "CRUISING"
        self.locked_obstacle = None
        self.log("[phase] CRUISING")

        def found(analysis, _missing, _elapsed):
            return analysis.path_blocked

        self._run_visual_motion(
            "forward",
            self.parameters["motion"]["linear_speed"],
            self.parameters["control"]["cruise_timeout_seconds"],
            "avoidance_cruise",
            found,
        )
        self._activate_locked_obstacle(self.last_analysis)

    def _phase_turn_to_tangent(self):
        self.state = "TURN_TO_TANGENT"
        settings = self.parameters["motion"]
        corridor = float(self.parameters["perception"]["center_corridor_width_norm"])
        margin = float(self.parameters["perception"]["tangent_margin_norm"])
        needed = max(1, int(self.parameters["control"].get("tangent_safe_frames", 3)))
        safe_streak = [0]
        tracked_bbox = [self.locked_obstacle["bbox"]]
        self.log(
            "[phase] TURN_TO_TANGENT direction={} locked_bbox={}".format(
                self.selected_side, tracked_bbox[0]
            )
        )

        def tangent_reached(analysis, _missing, _elapsed):
            if getattr(analysis, "status", None) == "IGNORED_WALL":
                safe_streak[0] += 1
                self.log(
                    "[tangent] wall-as-clear safe_streak={}/{}".format(
                        safe_streak[0], needed
                    )
                )
                return safe_streak[0] >= needed
            matched_bbox = self._find_tracked_bbox(analysis, tracked_bbox[0])
            if matched_bbox is not None:
                x, _y, width, _height = matched_bbox
                image_width = analysis.mask.shape[1]
                left = float(x) / float(image_width)
                right = float(x + width) / float(image_width)
                tracked_bbox[0] = matched_bbox
                self.locked_obstacle["tracked_bbox"] = matched_bbox
                if self.selected_side == "left":
                    geometry_safe = left >= 0.5 + corridor / 2.0 + margin
                else:
                    geometry_safe = right <= 0.5 - corridor / 2.0 - margin
            else:
                geometry_safe = True
                self.log("[tangent] locked bbox absent; frame counts as outside collision corridor")
            safe_streak[0] = safe_streak[0] + 1 if geometry_safe else 0
            self.log(
                "[tangent] safe_streak={}/{} geometry_safe={}".format(
                    safe_streak[0], needed, geometry_safe
                )
            )
            return safe_streak[0] >= needed

        self.tangent_turn_seconds = self._run_visual_motion(
            self.selected_side,
            settings["turn_speed"],
            settings["maximum_tangent_turn_seconds"],
            "turn_to_tangent",
            tangent_reached,
            minimum_seconds=settings["minimum_tangent_turn_seconds"],
            ignore_obstacle_failures=True,
        )
        self.estimated_heading_rad = self._turn_response_model().angle_radians(
            self.selected_side,
            settings["turn_speed"],
            self.tangent_turn_seconds,
        )
        self.state = "OFFSET_OUT"
        self.log("[phase] tangent reached seconds={:.3f}".format(self.tangent_turn_seconds))

    def _opposite(self, side=None):
        selected = self.selected_side if side is None else side
        return "right" if selected == "left" else "left"

    def _turn_response_model(self):
        if not hasattr(self, "turn_response"):
            self.turn_response = TurnResponseModel(
                None,
                self.parameters["motion"].get("angular_radians_per_speed_second", 3.14159265359),
            )
        return self.turn_response

    def _segment_minimum_seconds(self, configured_minimum):
        return max(
            float(configured_minimum),
            float(getattr(self, "depth_scaled_segment_seconds", 0.0)),
        )

    def _visual_segment_timeout(self, configured_maximum, minimum_seconds):
        control = self.parameters["control"]
        decision_seconds = (
            max(1, int(control.get("blocked_failure_frames", 5)))
            / max(0.2, float(control["control_fps"]))
        )
        correction_grace = max(
            float(control.get("timed_motion_grace_seconds", 1.0)),
            decision_seconds,
        )
        return max(float(configured_maximum), float(minimum_seconds) + correction_grace)

    def _phase_offset_out(self):
        self.state = "OFFSET_OUT"
        self.log("[phase] OFFSET_OUT")
        settings = self.parameters["motion"]

        def center_clear(analysis, _missing, _elapsed):
            return not analysis.path_blocked

        minimum_seconds = self._segment_minimum_seconds(settings["minimum_offset_seconds"])
        timeout_seconds = self._visual_segment_timeout(
            settings["maximum_offset_seconds"], minimum_seconds
        )
        self.log(
            "[phase] OFFSET_OUT minimum_seconds={:.3f} timeout_seconds={:.3f}".format(
                minimum_seconds, timeout_seconds
            )
        )
        self.offset_out_seconds = self._run_visual_motion(
            "forward",
            settings["linear_speed"],
            timeout_seconds,
            "offset_out",
            center_clear,
            minimum_seconds=minimum_seconds,
            confirm_new_obstacles=True,
        )
        segment = {
            "turn_seconds": float(self.tangent_turn_seconds),
            "offset_seconds": float(self.offset_out_seconds),
            "side": self.selected_side,
        }
        self.offset_segments.append(segment)
        if self.return_segments is not None:
            self.return_segments.insert(0, dict(segment))
        distance = (
            float(settings["linear_speed"])
            * float(self.offset_out_seconds)
            * float(settings["linear_meters_per_speed_second"])
        )
        self.estimated_lateral_offset_m += distance * np.sin(self.estimated_heading_rad)
        if getattr(self, "motion_interrupted_by_obstacle", None) is not None:
            self.lateral_correction_count = 0
            self._activate_locked_obstacle(self.motion_interrupted_by_obstacle)
            self.log(
                "[phase] OFFSET_OUT interrupted after {:.3f}s; "
                "partial segment stored and nested avoidance started".format(
                    self.offset_out_seconds
                )
            )
            return
        self.log(
            "[phase] offset clear seconds={:.3f} segments={} estimated_lateral_m={:.3f}".format(
                self.offset_out_seconds, len(self.offset_segments), self.estimated_lateral_offset_m
            )
        )
        self.state = "TURN_PARALLEL"

    def _phase_turn_parallel(self):
        self.state = "TURN_PARALLEL"
        self.log("[phase] TURN_PARALLEL")
        opposite = self._opposite()
        turn_seconds = self._turn_response_model().matching_seconds(
            self.selected_side,
            opposite,
            self.parameters["motion"]["turn_speed"],
            self.tangent_turn_seconds,
        )
        self._run_timed_motion(opposite, self.parameters["motion"]["turn_speed"], turn_seconds, "turn_parallel")
        self.estimated_heading_rad = 0.0
        if self._verify_clear_while_stopped("turn_parallel"):
            self.state = "PASSING"
            return
        limit = max(1, int(self.parameters["control"].get("maximum_lateral_corrections", 3)))
        if self.lateral_correction_count >= limit:
            raise RuntimeError(
                "turn_parallel remained blocked after {} lateral corrections".format(limit)
            )
        self.lateral_correction_count += 1
        self.state = "DETECTED"
        self.log(
            "[correction] turn_parallel blocked; repeat tangent/offset correction={}/{}".format(
                self.lateral_correction_count, limit
            )
        )

    def _phase_passing(self):
        self.state = "PASSING"
        self.log("[phase] PASSING")
        settings = self.parameters["motion"]

        def target_duration_reached(_analysis, _missing, _elapsed):
            return True

        minimum_seconds = max(
            float(settings["pass_min_seconds"]),
            float(getattr(self, "pass_depth_scaled_seconds", 0.0)),
        )
        timeout_seconds = self._visual_segment_timeout(
            settings["pass_max_seconds"], minimum_seconds
        )
        self.log(
            "[phase] PASSING minimum_seconds={:.3f} timeout_seconds={:.3f}".format(
                minimum_seconds, timeout_seconds
            )
        )
        actual_seconds = self._run_visual_motion(
            "forward",
            settings["linear_speed"],
            timeout_seconds,
            "pass_obstacle",
            target_duration_reached,
            minimum_seconds=minimum_seconds,
            confirm_new_obstacles=True,
        )
        if getattr(self, "motion_interrupted_by_obstacle", None) is not None:
            self.lateral_correction_count = 0
            self._activate_locked_obstacle(self.motion_interrupted_by_obstacle)
            self.log("[phase] PASSING interrupted; starting nested avoidance")
            return
        self.log(
            "[motion] pass_obstacle complete target_seconds={:.3f} "
            "actual_seconds={:.3f} delta_seconds={:.3f}".format(
                minimum_seconds,
                float(actual_seconds),
                float(actual_seconds) - minimum_seconds,
            )
        )
        self.state = "TURN_TO_LINE"

    def _phase_turn_to_line(self):
        self.state = "TURN_TO_LINE"
        self.log("[phase] TURN_TO_LINE")
        if self.return_segments is None:
            self.return_segments = list(reversed(self.offset_segments))
        if not self.return_segments:
            raise RuntimeError("offset return has no recorded outbound segments")
        self.active_return_segment = self.return_segments.pop(0)
        turn_seconds = float(self.active_return_segment["turn_seconds"])
        segment_side = self.active_return_segment.get("side", self.selected_side)
        return_direction = self._opposite(segment_side)
        return_turn_seconds = self._turn_response_model().matching_seconds(
            segment_side,
            return_direction,
            self.parameters["motion"]["turn_speed"],
            turn_seconds,
        )
        self._run_timed_motion(
            return_direction,
            self.parameters["motion"]["turn_speed"],
            return_turn_seconds,
            "turn_to_line",
        )
        self.estimated_heading_rad = self._turn_response_model().angle_radians(
            return_direction,
            self.parameters["motion"]["turn_speed"],
            return_turn_seconds,
        )
        self.state = "OFFSET_RETURN"

    def _phase_offset_return(self):
        self.state = "OFFSET_RETURN"
        self.log("[phase] OFFSET_RETURN")
        if self.active_return_segment is None:
            raise RuntimeError("offset return has no active outbound segment")
        if self.active_return_segment.get("exact_seconds", False):
            return_seconds = float(self.active_return_segment["offset_seconds"])
        else:
            return_seconds = max(
                float(self.active_return_segment["offset_seconds"]),
                float(getattr(self, "depth_scaled_segment_seconds", 0.0)),
            )
        self.log(
            "[phase] OFFSET_RETURN seconds={:.3f} segment_base_seconds={:.3f}".format(
                return_seconds,
                float(getattr(self, "depth_scaled_segment_seconds", 0.0)),
            )
        )
        def target_duration_reached(_analysis, _missing, _elapsed):
            return True

        actual_seconds = self._run_visual_motion(
            "forward",
            self.parameters["motion"]["linear_speed"],
            self._visual_segment_timeout(return_seconds, return_seconds),
            "offset_return",
            target_duration_reached,
            minimum_seconds=return_seconds,
            confirm_new_obstacles=True,
        )
        if getattr(self, "motion_interrupted_by_obstacle", None) is not None:
            remaining_seconds = max(0.0, return_seconds - float(actual_seconds))
            interrupted_side = self.active_return_segment.get("side", self.selected_side)
            if remaining_seconds > 0.01:
                self.return_segments.insert(0, {
                    "turn_seconds": 0.0,
                    "offset_seconds": remaining_seconds,
                    "side": interrupted_side,
                    "exact_seconds": True,
                })
            self.active_return_segment = None
            self.lateral_correction_count = 0
            self._activate_locked_obstacle(self.motion_interrupted_by_obstacle)
            self.log(
                "[phase] OFFSET_RETURN interrupted; remaining_seconds={:.3f} "
                "stored for route resume".format(remaining_seconds)
            )
            return
        self.estimated_lateral_offset_m = 0.0
        self.state = "RESTORE_HEADING"

    def _phase_restore_heading(self):
        self.state = "RESTORE_HEADING"
        self.log("[phase] RESTORE_HEADING")
        if self.active_return_segment is None:
            raise RuntimeError("heading restore has no active outbound segment")
        segment_side = self.active_return_segment.get("side", self.selected_side)
        self._run_timed_motion(
            segment_side,
            self.parameters["motion"]["turn_speed"],
            float(self.active_return_segment["turn_seconds"]),
            "restore_heading",
        )
        self.estimated_heading_rad = 0.0
        self.active_return_segment = None
        if self.return_segments:
            self.state = "TURN_TO_LINE"
            self.log("[phase] returning next stored offset segment")
        else:
            self.state = "VERIFY_CLEAR"

    def _phase_verify(self):
        self.state = "VERIFY_CLEAR"
        self.log("[phase] VERIFY_CLEAR")
        self._run_timed_motion(
            "forward",
            self.parameters["motion"]["linear_speed"],
            self.parameters["motion"]["verification_forward_seconds"],
            "verify_clear",
            require_clear=True,
        )
        self.state = "DONE"
        self.log("[avoidance] DONE route and heading restored by symmetric timing")
        self._close_writer()

    def run_next_phase(self):
        phases = {
            "READY": self._phase_cruise,
            "DETECTED": self._phase_turn_to_tangent,
            "OFFSET_OUT": self._phase_offset_out,
            "TURN_PARALLEL": self._phase_turn_parallel,
            "PASSING": self._phase_passing,
            "TURN_TO_LINE": self._phase_turn_to_line,
            "OFFSET_RETURN": self._phase_offset_return,
            "RESTORE_HEADING": self._phase_restore_heading,
            "VERIFY_CLEAR": self._phase_verify,
        }
        if self.state not in phases:
            raise RuntimeError("state {} has no next phase; reset the session if needed".format(self.state))
        phases[self.state]()
        return self.state

    def run_full(self):
        self.start_camera()
        started = time.time()
        while self.state not in ("DONE", "FAILED", "STOPPED"):
            if time.time() - started > float(self.parameters["safety"]["maximum_total_seconds"]):
                raise RuntimeError("maximum total runtime exceeded")
            self.run_next_phase()
        return self.state == "DONE"

    def start_job(self, operation, name, replace_live=False):
        with self.worker_lock:
            if self.worker is not None and self.worker.is_alive():
                if replace_live and self.worker_name == "live_monitor":
                    self.log("[job] stopping live monitor before {}".format(name))
                    self.stop_event.set()
                    self.base.stop()
                    self.worker.join(5.0)
                    if self.worker is not None and self.worker.is_alive():
                        raise RuntimeError("live monitor is still stopping; click STOP BASE and wait")
                else:
                    raise RuntimeError(
                        "another avoidance job is still running: {}".format(self.worker_name or "unknown")
                    )
            self.stop_event.clear()
            self.worker_name = name

            def target():
                try:
                    self.log("[job] start {}".format(name))
                    result = operation()
                    self.log("[job] finish {} result={}".format(name, result))
                except Exception as exc:
                    self.base.stop()
                    self.state = "STOPPED" if self.stop_event.is_set() else "FAILED"
                    self.log("[error] {}: {}".format(name, exc))
                    self.log(traceback.format_exc())
                    self._close_writer()
                finally:
                    self.base.stop()
                    if self.stop_event.is_set():
                        self._close_writer()
                    self.worker = None
                    self.worker_name = None

            self.worker = threading.Thread(target=target)
            self.worker.daemon = True
            self.worker.start()
            return True

    def start_live_monitor(self):
        def monitor():
            while not self.stop_event.is_set():
                self.sample()
                self._sleep_control()
            return True
        return self.start_job(monitor, "live_monitor")


def build_ui(parameters_path=DEFAULT_PARAMETERS):
    import ipywidgets as widgets
    from IPython.display import FileLink, Javascript, display

    output = widgets.Output(layout=widgets.Layout(border="1px solid #bbb", height="380px", overflow_y="auto"))
    image = widgets.Image(format="jpeg", layout=widgets.Layout(width="640px"))
    base_real = widgets.Checkbox(value=False, description="base_real")
    camera_real = widgets.Checkbox(value=True, description="camera_real")
    live = widgets.Checkbox(value=True, description="live")
    record = widgets.Checkbox(value=False, description="record")
    geometry_hud = widgets.Checkbox(value=True, description="Geometry HUD")
    text_hud = widgets.Checkbox(value=True, description="Text HUD")
    session = {"value": None}

    def append_log(message):
        output.append_stdout(str(message) + "\n")

    def current(create=True):
        if session["value"] is None and create:
            value = ObstacleAvoidanceSession(parameters_path, base_real.value, camera_real.value)
            value.log_callback = append_log
            value.frame_callback = lambda data: setattr(image, "value", data)
            session["value"] = value
        value = session["value"]
        if value is not None:
            value.configure_media(
                live.value,
                record.value,
                geometry_hud.value,
                text_hud.value,
            )
        return value

    def guarded(fn):
        def wrapped(_=None):
            try:
                result = fn()
                append_log("[result] {}".format(result))
            except Exception as exc:
                output.append_stderr("[error] {}\n".format(exc))
                output.append_stderr(traceback.format_exc())
        return wrapped

    def reset_session():
        value = session["value"]
        if value is not None:
            value.close()
        session["value"] = None
        return "session reset"

    def stop_base():
        value = current(False)
        if value is not None:
            value.stop_base()
        return "base stop requested"

    def stop_release():
        value = current(False)
        if value is not None:
            return value.release_camera()
        return {"released": True, "reason": "no active session"}

    def recording_link():
        value = current(False)
        if value is None or not value.recording_path or not os.path.isfile(value.recording_path):
            return "no recording available"
        relative = os.path.relpath(value.recording_path, os.getcwd())
        display(FileLink(relative, result_html_prefix="Download recording: "))
        return value.recording_path

    buttons = {
        "reload": widgets.Button(description="Reload JSON", button_style="info"),
        "load": widgets.Button(description="Load Camera + Depth"),
        "analyze": widgets.Button(description="Analyze Frame"),
        "live": widgets.Button(description="Start Live"),
        "next": widgets.Button(description="Run Next Phase"),
        "full": widgets.Button(description="Run Full Bypass", button_style="success"),
        "stop": widgets.Button(description="STOP BASE", button_style="danger"),
        "stop_media": widgets.Button(description="Stop Media"),
        "release": widgets.Button(description="Stop + Release Camera"),
        "reset": widgets.Button(description="Reset Session"),
        "link": widgets.Button(description="Recording Link"),
        "clear": widgets.Button(description="Clear Log"),
    }
    buttons["reload"].on_click(guarded(lambda: current().reload_parameters()))
    buttons["load"].on_click(guarded(lambda: current().start_camera()))
    buttons["analyze"].on_click(guarded(lambda: current().analyze_frame()))
    buttons["live"].on_click(guarded(lambda: current().start_live_monitor()))
    buttons["next"].on_click(guarded(
        lambda: current().start_job(current().run_next_phase, "next_phase", replace_live=True)
    ))
    buttons["full"].on_click(guarded(
        lambda: current().start_job(current().run_full, "full_bypass", replace_live=True)
    ))
    buttons["stop"].on_click(guarded(stop_base))
    buttons["stop_media"].on_click(guarded(lambda: current().stop_media()))
    buttons["release"].on_click(guarded(stop_release))
    buttons["reset"].on_click(guarded(reset_session))
    buttons["link"].on_click(guarded(recording_link))
    buttons["clear"].on_click(lambda _: output.clear_output())

    panel = widgets.VBox([
        widgets.HTML("<b>Simplified TangentBug obstacle tuning</b>"),
        widgets.HBox([base_real, camera_real, live, record]),
        widgets.HBox([geometry_hud, text_hud]),
        widgets.HBox([buttons["reload"], buttons["load"], buttons["analyze"], buttons["live"]]),
        widgets.HBox([buttons["next"], buttons["full"], buttons["stop"]]),
        widgets.HBox([buttons["stop_media"], buttons["release"], buttons["reset"], buttons["link"], buttons["clear"]]),
        image,
        output,
    ])
    display(panel)
    display(Javascript(
        "setTimeout(function(){var x=document.querySelectorAll('.widget-output');"
        "for(var i=0;i<x.length;i++){var e=x[i];if(e.__avoidScroll)continue;"
        "var o=new MutationObserver(function(m){var t=m[0].target.closest('.widget-output');"
        "if(t)t.scrollTop=t.scrollHeight;});o.observe(e,{childList:true,subtree:true});"
        "e.__avoidScroll=o;}},100);"
    ))
    append_log("[avoidance] ready; base_real is disabled by default")
    return panel
