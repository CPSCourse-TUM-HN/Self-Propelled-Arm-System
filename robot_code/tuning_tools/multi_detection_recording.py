from __future__ import print_function

import csv
import json
import math
import os
import threading
import time
import traceback

import cv2
import ipywidgets as widgets
from IPython.display import FileLink, display

from demo_core import DemoStateMachine, MissionState, Pose2D, TargetType, load_config
from demo_core.perception import DepthSensor
from demo_core.tangentbug import DepthTangentBugPlanner
from tuning_tools.diagnostic_tools import CandidateCountTracker, candidate_bbox_metrics
from tuning_tools.obstacle_avoidance_tuning import load_avoidance_parameters


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _session_directory(category):
    path = os.path.join(PROJECT_ROOT, "logs", category, time.strftime("%Y%m%d_%H%M%S"))
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


class RecordingPanelBase(object):
    """Shared live/recording lifecycle for read-only multi-object diagnostics."""

    category = "multi_detection"

    def __init__(self):
        self.camera_real = widgets.Checkbox(value=False, description="camera_real")
        self.live_stream = widgets.Checkbox(value=True, description="live_stream")
        self.record_camera = widgets.Checkbox(value=False, description="record_camera")
        self.sample_fps = widgets.FloatText(value=3.0, description="sample_fps")
        self.image = widgets.Image(format="jpeg", width=640, height=480)
        self.output = widgets.Output(layout={"border": "1px solid #bbb", "height": "300px", "overflow_y": "auto"})
        self.session_dir = _session_directory(self.category)
        self.thread = None
        self.running = False
        self.last_recording_path = None
        self.last_csv_path = None

    def _log(self, text, error=False):
        append = self.output.append_stderr if error else self.output.append_stdout
        append(str(text).rstrip() + "\n")

    def start_hardware(self):
        raise NotImplementedError

    def process_frame(self):
        raise NotImplementedError

    def stop_hardware(self):
        raise NotImplementedError

    def _loop(self):
        writer = csv_stream = csv_writer = None
        try:
            while self.running:
                canvas, rows = self.process_frame()
                if self.live_stream.value:
                    ok, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                    if ok:
                        self.image.value = encoded.tobytes()
                if self.record_camera.value and writer is None:
                    stamp = str(int(time.time() * 1000))
                    self.last_recording_path = os.path.join(self.session_dir, "recording_{}.avi".format(stamp))
                    self.last_csv_path = os.path.join(self.session_dir, "recording_{}.csv".format(stamp))
                    writer = cv2.VideoWriter(self.last_recording_path, cv2.VideoWriter_fourcc(*"MJPG"), max(1.0, float(self.sample_fps.value)), (640, 480))
                    if not writer.isOpened():
                        writer.release()
                        raise RuntimeError("recorder could not open {}".format(self.last_recording_path))
                    csv_stream = open(self.last_csv_path, "w", newline="")
                    csv_writer = csv.DictWriter(csv_stream, fieldnames=list(rows[0].keys()), extrasaction="ignore")
                    csv_writer.writeheader()
                    self._log("[recording] started {} csv={}".format(self.last_recording_path, self.last_csv_path))
                if writer is not None:
                    if self.record_camera.value:
                        writer.write(canvas)
                        csv_writer.writerows(rows)
                        csv_stream.flush()
                    else:
                        writer.release(); writer = None
                        csv_stream.close(); csv_stream = csv_writer = None
                        self._log("[recording] stopped {}".format(self.last_recording_path))
                time.sleep(1.0 / max(0.2, float(self.sample_fps.value)))
        except Exception as exc:
            self._log("[monitor] error: {}".format(exc), True)
            self._log(traceback.format_exc(), True)
        finally:
            if writer is not None:
                writer.release()
            if csv_stream is not None:
                csv_stream.close()
            self.running = False
            self.thread = None
            self._log("[monitor] stopped")

    def start_monitor(self):
        if self.thread is not None and self.thread.is_alive():
            return {"started": False, "reason": "monitor already running"}
        if not self.live_stream.value and not self.record_camera.value:
            return {"started": False, "reason": "enable live_stream or record_camera"}
        self.start_hardware()
        self.running = True
        self.thread = threading.Thread(target=self._loop)
        self.thread.daemon = True
        self.thread.start()
        return {"started": True, "session_dir": self.session_dir}

    def stop_monitor(self, wait_seconds=5.0):
        self.running = False
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(float(wait_seconds))
        alive = thread is not None and thread.is_alive()
        if not alive:
            self.thread = None
        return {"stopped": not alive, "recording": self.last_recording_path, "csv": self.last_csv_path}

    def release(self):
        result = self.stop_monitor()
        self.stop_hardware()
        return result

    def clear_log(self):
        self.output.clear_output()
        return True

    def recording_links(self):
        available = False
        for label, path in (("Video", self.last_recording_path), ("CSV", self.last_csv_path)):
            if path and os.path.isfile(path):
                display(FileLink(os.path.relpath(path, os.getcwd()), result_html_prefix="{}: ".format(label)))
                available = True
        return {"available": available, "video": self.last_recording_path, "csv": self.last_csv_path}

    def _button(self, label, callback, style=""):
        button = widgets.Button(description=label, button_style=style)
        def guarded(_):
            try:
                self._log("[result] {}".format(callback()))
            except Exception as exc:
                self._log("[error] {}".format(exc), True)
                self._log(traceback.format_exc(), True)
        button.on_click(guarded)
        return button

    def controls(self):
        return widgets.HBox([
            self._button("Start Monitor", self.start_monitor, "success"),
            self._button("Stop Monitor", self.stop_monitor, "warning"),
            self._button("STOP + Release", self.release, "danger"),
            self._button("Recording Links", self.recording_links),
            self._button("Clear Log", self.clear_log),
        ])


class DualCanMapRecordingPanel(RecordingPanelBase):
    category = "dual_can_vague_map"

    def __init__(self):
        RecordingPanelBase.__init__(self)
        self.runtime = None
        self.frame_index = 0
        self.pose_x = widgets.FloatText(value=0.0, description="robot_x_m")
        self.pose_y = widgets.FloatText(value=0.0, description="robot_y_m")
        self.pose_heading_deg = widgets.FloatText(value=0.0, description="heading_deg")
        self.merge_radius_m = widgets.FloatText(value=0.3, description="merge_radius_m")
        self.show_vague_map = widgets.Checkbox(value=False, description="show_vague_map")
        self.maximum_visible_cans = 0

    def start_hardware(self):
        if self.runtime is not None:
            return
        if not self.camera_real.value:
            raise RuntimeError("enable camera_real before starting")
        config = load_config(overrides={"runtime": {"dry_run": {"camera": False, "base": True, "arm": True}}, "camera": {"depth_enabled": True}})
        self.runtime = DemoStateMachine(config)
        self.runtime.services.depth.start()
        self.runtime.services.can_detector.load()
        self.reset_map()
        self._log("[dual-can] camera + DepthNet + can detector ready; base/arm dry-run")

    def reset_map(self):
        if self.runtime is None:
            return {"reset": False, "reason": "hardware not started"}
        vague_map = self.runtime.vague_map
        vague_map.known_cans.clear()
        vague_map.selected_can_id = None
        vague_map._next_can_id = 1
        vague_map.settings["can_merge_radius_m"] = float(self.merge_radius_m.value)
        self.maximum_visible_cans = 0
        vague_map.set_robot_pose(Pose2D(self.pose_x.value, self.pose_y.value, math.radians(float(self.pose_heading_deg.value))), "dual_can_test")
        self.runtime.context.grabbed = True
        self.runtime.context.target_type = TargetType.BIN
        self.runtime.state = MissionState.SEARCHING
        return {"reset": True, "pose": vague_map.robot_pose.as_dict()}

    def _map_point(self, position, bounds, rectangle):
        left, top, right, bottom = rectangle
        x = (float(position["x_m"]) - float(bounds["min_x"])) / (float(bounds["max_x"]) - float(bounds["min_x"]))
        y = (float(position["y_m"]) - float(bounds["min_y"])) / (float(bounds["max_y"]) - float(bounds["min_y"]))
        return left + int(x * (right - left)), bottom - int(y * (bottom - top))

    def _draw_map(self, canvas, snapshot):
        bounds = self.runtime.config.get("vague_map.bounds_m")
        rectangle = (430, 275, 635, 475)
        cv2.rectangle(canvas, rectangle[:2], rectangle[2:], (25, 25, 25), -1)
        cv2.rectangle(canvas, rectangle[:2], rectangle[2:], (220, 220, 220), 1)
        cv2.circle(canvas, self._map_point(snapshot["robot_pose"], bounds, rectangle), 5, (255, 180, 0), -1)
        for item in snapshot["known_cans"]:
            center = self._map_point(item["position"], bounds, rectangle)
            cv2.circle(canvas, center, 6, (0, 255, 0), -1)
            cv2.putText(canvas, "C{}".format(item["can_id"]), (center[0] + 7, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(canvas, "VAGUE MAP cans={}".format(len(snapshot["known_cans"])), (435, 293), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def process_frame(self):
        frame = self.runtime.services.depth.read_frame()
        if frame is None:
            raise RuntimeError("camera frame unavailable")
        detections = self.runtime.services.can_detector.detect_all(frame)
        mapped = self.runtime._map_can_detections(
            frame,
            detections,
            confirmation_frames=self.runtime._incidental_can_confirmation_frames(),
        )
        snapshot = self.runtime.vague_map.snapshot()
        self.maximum_visible_cans = max(self.maximum_visible_cans, len(detections))
        display_frame = self.runtime.services.can_detector.display_frame(frame)
        canvas = cv2.resize(display_frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        sx, sy = 640.0 / frame.shape[1], 480.0 / frame.shape[0]
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
            cv2.rectangle(canvas, p1, p2, (0, 255, 255), 2)
            cv2.putText(canvas, "CAN #{}".format(index + 1), (p1[0], max(18, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        if self.show_vague_map.value:
            self._draw_map(canvas, snapshot)
        cv2.rectangle(canvas, (0, 0), (640, 54), (20, 20, 20), -1)
        cv2.putText(canvas, "CAN DETECT HUD | visible={} max_visible={}".format(len(detections), self.maximum_visible_cans), (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 0) if detections else (160, 160, 160), 1)
        cv2.putText(canvas, "CARRYING | mapped_now={} remembered={} map_view={} {}".format(len(mapped), len(snapshot["known_cans"]), "ON" if self.show_vague_map.value else "OFF", "REC" if self.record_camera.value else "LIVE"), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        common = {"timestamp": time.time(), "frame_index": self.frame_index, "detection_count": len(detections), "maximum_visible_cans": self.maximum_visible_cans, "mapped_this_frame": len(mapped), "known_can_count": len(snapshot["known_cans"]), "show_vague_map": bool(self.show_vague_map.value), "merge_radius_m": float(self.merge_radius_m.value), "known_cans_json": json.dumps(snapshot["known_cans"], sort_keys=True)}
        rows = []
        for index, detection in enumerate(detections):
            row = dict(common)
            row.update({"detection_index": index, "confidence": detection.get("confidence"), "center_x": detection.get("center_x"), "center_y": detection.get("center_y"), "error_x": detection.get("error_x")})
            rows.append(row)
        if not rows:
            row = dict(common); row.update({"detection_index": "", "confidence": "", "center_x": "", "center_y": "", "error_x": ""}); rows.append(row)
        self.frame_index += 1
        return canvas, rows

    def stop_hardware(self):
        if self.runtime is not None:
            self.runtime.services.depth.stop()
            self.runtime.services.can_detector.reset()
            self.runtime = None
        self._log("[dual-can] hardware released")

    def ui(self):
        return widgets.VBox([
            widgets.HTML("<b>Dual-can carrying-map recording</b>: base and arm are always dry-run."),
            widgets.HBox([self.camera_real, self.live_stream, self.record_camera, self.sample_fps]),
            widgets.HBox([self.pose_x, self.pose_y, self.pose_heading_deg]),
            widgets.HBox([self.show_vague_map, self.merge_radius_m, self._button("Reset Vague Map", self.reset_map, "info")]),
            self.controls(), self.image, self.output,
        ])


class DualObstacleRecordingPanel(RecordingPanelBase):
    category = "dual_obstacle_detection"

    def __init__(self):
        RecordingPanelBase.__init__(self)
        self.depth = self.planner = None
        self.tracker = CandidateCountTracker(2)
        self.frame_index = 0
        self.scenario = widgets.Text(value="clearly_separated", description="scenario")

    def start_hardware(self):
        if self.depth is not None:
            return
        if not self.camera_real.value:
            raise RuntimeError("enable camera_real before starting DepthNet")
        params = load_avoidance_parameters()
        config = load_config(overrides={"runtime": {"dry_run": {"camera": False}}, "camera": {"depth_enabled": True, "width": params["camera"]["width"], "height": params["camera"]["height"], "depth_network": params["camera"]["depth_network"]}})
        self.depth = DepthSensor(config)
        self.depth.start()
        self.planner = DepthTangentBugPlanner(params["perception"])
        self._log("[dual-obstacle] camera + DepthNet ready; no base controller constructed")

    def reset_stats(self):
        self.tracker = CandidateCountTracker(2)
        self.frame_index = 0
        return {"reset": True}

    def process_frame(self):
        frame = self.depth.read_frame()
        frame = self.depth.depth_input_frame(frame)
        depth_map = self.depth.depth_map_frame(frame, frame_space=self.depth.depth_frame_space)
        if frame is None or depth_map is None:
            raise RuntimeError("camera/depth frame unavailable")
        analysis = self.planner.analyze_obstacle(depth_map)
        candidates = candidate_bbox_metrics(analysis, depth_map)
        summary = self.tracker.update_candidates(candidates)
        canvas = self.planner.draw_obstacle_debug(frame, analysis, state="COUNT={}".format(len(candidates)), draw_geometry=True, draw_text=True)
        for item in candidates:
            x, y, width, height = item["bbox"]
            cv2.rectangle(canvas, (x, y), (x + width, y + height), (255, 0, 255), 2)
            cv2.putText(canvas, "#{}".format(item["candidate_index"]), (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
        canvas = cv2.resize(canvas, (640, 480), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(canvas, (0, 0), (640, 48), (20, 20, 20), -1)
        cv2.putText(canvas, "scenario={} candidates={} dual={:.2f} merge={:.2f} split={:.2f}".format(self.scenario.value, len(candidates), summary["expected_count_rate"], summary["merge_rate"], summary["false_split_rate"]), (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        common = {"timestamp": time.time(), "frame_index": self.frame_index, "scenario": self.scenario.value, "candidate_count": len(candidates), "dual_detection_rate": summary["expected_count_rate"], "merge_rate": summary["merge_rate"], "false_split_rate": summary["false_split_rate"], "identity_stability_rate": summary["identity_stability_rate"]}
        rows = []
        for item in candidates:
            x, y, width, height = item["bbox"]
            row = dict(common); row.update({"candidate_index": item["candidate_index"], "bbox_x": x, "bbox_y": y, "bbox_width": width, "bbox_height": height, "center_x_norm": item["center_x_norm"], "center_y_norm": item["center_y_norm"], "area_norm": item["area_norm"], "median_depth": item["median_depth"]}); rows.append(row)
        if not rows:
            row = dict(common); row.update({"candidate_index": "", "bbox_x": "", "bbox_y": "", "bbox_width": "", "bbox_height": "", "center_x_norm": "", "center_y_norm": "", "area_norm": "", "median_depth": ""}); rows.append(row)
        self.frame_index += 1
        return canvas, rows

    def stop_hardware(self):
        if self.depth is not None:
            self.depth.stop()
            self.depth = self.planner = None
        self._log("[dual-obstacle] hardware released")

    def ui(self):
        return widgets.VBox([
            widgets.HTML("<b>Dual-obstacle live/recording test</b>: camera + DepthNet only; no base controller."),
            widgets.HBox([self.camera_real, self.live_stream, self.record_camera, self.sample_fps, self.scenario]),
            widgets.HBox([self._button("Reset Statistics", self.reset_stats, "info")]),
            self.controls(), self.image, self.output,
        ])


def build_dual_can_map_ui():
    return DualCanMapRecordingPanel()


def build_dual_obstacle_ui():
    return DualObstacleRecordingPanel()
