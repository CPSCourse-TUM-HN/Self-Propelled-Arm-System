from __future__ import print_function

import csv
import os
import threading
import time
import traceback

import cv2

from demo_core import load_config
from demo_core.navigation import APPROACH_STOP_MODES, evaluate_approach_stop
from demo_core.perception import CanDetector, DepthSensor
from demo_core.robot_control import BaseController


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def _finite_text(value, digits=3):
    return "n/a" if value is None else ("{:.%df}" % int(digits)).format(float(value))


class CanStableStopSession(object):
    """Isolated pre-arm-down can stop tuning with optional real base motion."""

    CSV_FIELDS = (
        "timestamp", "record_type", "run_id", "ideal_stop_distance_cm",
        "measured_stop_distance_cm", "mode", "found", "confidence", "error_x",
        "bbox_height_norm", "bbox_height_threshold", "bbox_reached",
        "depth_raw", "depth_raw_threshold", "depth_reached", "stop_reached",
        "base_real", "base_active", "result",
    )

    def __init__(self, base_real=False, camera_real=True, mode="bbox_height",
                 bbox_height_threshold=0.36, depth_raw_threshold=2.05,
                 speed=0.6, timeout_seconds=15.0, control_fps=2.0,
                 ideal_stop_distance_cm=10.0, live=True, record=False):
        self.mode = str(mode)
        self.bbox_height_threshold = float(bbox_height_threshold)
        self.depth_raw_threshold = float(depth_raw_threshold)
        self.speed = float(speed)
        self.timeout_seconds = float(timeout_seconds)
        self.control_fps = float(control_fps)
        self.ideal_stop_distance_cm = float(ideal_stop_distance_cm)
        self.base_real = bool(base_real)
        self.camera_real = bool(camera_real)
        self.live_enabled = bool(live)
        self.record_enabled = bool(record)
        self._validate_settings()
        self.config = load_config(overrides={
            "runtime": {"dry_run": {"base": not self.base_real, "camera": not self.camera_real}},
            "navigation": {"can": {"approach": {
                "stop": self.stop_settings(),
            }}},
        })
        self.base = BaseController(self.config)
        self.depth = DepthSensor(self.config)
        self.detector = CanDetector(self.config)
        self.stop_event = threading.Event()
        self.worker = None
        self.worker_name = None
        self.worker_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.camera_started = False
        self.frame_callback = None
        self.log_callback = None
        self.writer = None
        self.recording_path = None
        self.last_frame = None
        self.last_rendered = None
        self.last_sample = None
        self.last_result = None
        self.last_preview_at = 0.0
        self.last_record_at = 0.0
        self.run_id = None
        stamp = time.strftime("%Y%m%d_%H%M%S")
        logs_root = self.config.resolve_path(self.config.get("paths.logs", "logs"))
        self.output_directory = os.path.join(logs_root, "can_stable_stop", stamp)
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory)
        self.csv_path = os.path.join(self.output_directory, "samples.csv")
        with open(self.csv_path, "w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.CSV_FIELDS).writeheader()

    def _validate_settings(self):
        if self.mode not in APPROACH_STOP_MODES:
            raise ValueError("mode must be one of {}".format(APPROACH_STOP_MODES))
        if not 0.0 < self.bbox_height_threshold <= 1.0:
            raise ValueError("bbox_height_threshold must be in (0, 1]")
        if self.depth_raw_threshold <= 0.0:
            raise ValueError("depth_raw_threshold must be positive")
        if not 0.0 < self.speed <= 1.0:
            raise ValueError("speed must be in (0, 1]")
        if self.timeout_seconds <= 0.0 or self.control_fps <= 0.0:
            raise ValueError("timeout_seconds and control_fps must be positive")

    def stop_settings(self):
        return {
            "mode": self.mode,
            "bbox_height_threshold": self.bbox_height_threshold,
            "depth_raw_threshold": self.depth_raw_threshold,
        }

    def log(self, message):
        line = "{} {}".format(time.strftime("%H:%M:%S"), message)
        if self.log_callback is not None:
            self.log_callback(line)
        else:
            print(line)

    def start_camera(self):
        if self.camera_started:
            return
        self.depth.start(camera_only=False)
        self.detector.load()
        self.camera_started = True
        self.log("[camera] camera + DepthNet + can detector ready")

    def _writer_start(self, frame):
        if not self.record_enabled or self.writer is not None:
            return
        height, width = frame.shape[:2]
        self.recording_path = os.path.join(
            self.output_directory, "can_stable_stop_{}.avi".format(int(time.time() * 1000))
        )
        self.writer = cv2.VideoWriter(
            self.recording_path, cv2.VideoWriter_fourcc(*"MJPG"),
            max(1.0, self.control_fps), (int(width), int(height)),
        )
        if not self.writer.isOpened():
            self.writer.release()
            self.writer = None
            raise RuntimeError("could not open recording {}".format(self.recording_path))
        self.log("[record] started {}".format(self.recording_path))

    def stop_media(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.log("[record] finalized {}".format(self.recording_path))
        return self.recording_path

    def _append_csv(self, sample, record_type="sample", measured_stop_distance_cm=None):
        row = dict(sample or {})
        row.update({
            "timestamp": time.time(),
            "record_type": record_type,
            "run_id": self.run_id,
            "ideal_stop_distance_cm": self.ideal_stop_distance_cm,
            "measured_stop_distance_cm": measured_stop_distance_cm,
            "mode": self.mode,
            "base_real": self.base_real,
            "base_active": self.base.motion_active(),
            "result": self.last_result,
        })
        with open(self.csv_path, "a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.CSV_FIELDS, extrasaction="ignore")
            writer.writerow(row)

    def record_measurement(self, measured_stop_distance_cm):
        measured = float(measured_stop_distance_cm)
        self._append_csv(self.last_sample, "measurement", measured)
        self.log("[measurement] ideal_cm={:.2f} measured_cm={:.2f}".format(
            self.ideal_stop_distance_cm, measured
        ))
        return self.csv_path

    def _render(self, frame, observation, depth_stats, status):
        canvas = frame.copy()
        bbox = observation.get("bbox") if observation else None
        if bbox:
            left, top, right, bottom = [int(value) for value in bbox]
            cv2.rectangle(canvas, (left, top), (right, bottom), (0, 255, 0), 2)
        roi = (depth_stats or {}).get("roi")
        if roi:
            height, width = canvas.shape[:2]
            x1, y1, x2, y2 = roi
            cv2.rectangle(
                canvas, (int(x1 * width), int(y1 * height)),
                (int(x2 * width), int(y2 * height)), (255, 180, 0), 2,
            )
        stop_color = (0, 0, 255) if status["reached"] else (0, 220, 0)
        lines = (
            "CAN PRE-ARM STOP | mode={} | {}".format(self.mode, "STOP" if status["reached"] else "GO"),
            "bbox h={} >= {} : {}".format(
                _finite_text(status["bbox_height_norm"]),
                _finite_text(status["bbox_height_threshold"]), status["bbox_reached"]
            ),
            "center depth raw={} <= {} : {}".format(
                _finite_text(status["depth_raw"]),
                _finite_text(status["depth_raw_threshold"]), status["depth_reached"]
            ),
            "can found={} err_x={} base={} ideal={}cm".format(
                bool(observation and observation.get("found")),
                _finite_text(observation.get("error_x") if observation else None),
                "REAL" if self.base_real else "DRY", _finite_text(self.ideal_stop_distance_cm, 1),
            ),
        )
        for index, line in enumerate(lines):
            color = stop_color if index == 0 else (255, 255, 255)
            cv2.putText(canvas, line, (8, 22 + index * 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, color, 2, cv2.LINE_AA)
        if self.writer is not None or self.record_enabled:
            cv2.putText(canvas, "REC", (canvas.shape[1] - 48, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2, cv2.LINE_AA)
        return canvas

    def sample(self):
        self.start_camera()
        with self.inference_lock:
            frame = self.depth.read_frame()
            if frame is None:
                raise RuntimeError("camera frame unavailable")
            observation = self.detector.detect(frame)
            depth_stats = self.depth.observe_lens_center_frame(frame, report=False)
        depth_raw = None if not depth_stats else depth_stats.get("mean")
        status = evaluate_approach_stop(
            self.stop_settings(),
            bbox_height_norm=observation.get("bbox_height_norm") if observation else None,
            depth_raw=depth_raw,
        )
        sample = dict(status)
        sample.update({
            "found": bool(observation and observation.get("found")),
            "confidence": observation.get("confidence") if observation else None,
            "error_x": observation.get("error_x") if observation else None,
            "stop_reached": status["reached"],
        })
        display_frame = self.detector.display_frame(frame)
        rendered = self._render(display_frame, observation, depth_stats, status)
        self._writer_start(rendered)
        now = time.time()
        if self.writer is not None and now - self.last_record_at >= 1.0 / max(1.0, self.control_fps):
            self.writer.write(rendered)
            self.last_record_at = now
        if self.live_enabled and self.frame_callback is not None and now - self.last_preview_at >= 0.2:
            ok, encoded = cv2.imencode(".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok:
                self.frame_callback(encoded.tobytes())
            self.last_preview_at = now
        self.last_frame = display_frame
        self.last_rendered = rendered
        self.last_sample = sample
        self._append_csv(sample)
        return sample

    def run_trial(self):
        self.start_camera()
        self.stop_event.clear()
        self.run_id = time.strftime("trial_%Y%m%d_%H%M%S")
        self.last_result = "running"
        started = time.time()
        self.log("[trial] start mode={} bbox={} depth_raw={} speed={} base_real={}".format(
            self.mode, self.bbox_height_threshold, self.depth_raw_threshold,
            self.speed, self.base_real,
        ))
        try:
            while not self.stop_event.is_set():
                if time.time() - started >= self.timeout_seconds:
                    self.last_result = "timeout"
                    break
                sample = self.sample()
                if sample["stop_reached"]:
                    self.last_result = "threshold_reached"
                    break
                if not sample["found"]:
                    self.base.stop()
                    self.last_result = "waiting_for_can"
                elif abs(float(sample.get("error_x") or 0.0)) > float(
                        self.config.get("navigation.can.approach.steering_tolerance_norm", 0.12)):
                    self.base.stop()
                    self.last_result = "align_can_manually"
                else:
                    self.base.start_motion("forward", self.speed, "can_stable_stop_trial")
                    self.last_result = "approaching"
                time.sleep(1.0 / self.control_fps)
        finally:
            self.base.stop()
            if self.stop_event.is_set():
                self.last_result = "stopped"
            self._append_csv(self.last_sample, "result")
            self.log("[trial] result={}".format(self.last_result))
        return self.last_result

    def run_live(self):
        self.start_camera()
        self.stop_event.clear()
        self.last_result = "live"
        self.log("[live] started; base remains stopped")
        self.base.stop()
        try:
            while not self.stop_event.is_set():
                self.sample()
                time.sleep(1.0 / self.control_fps)
        finally:
            self.base.stop()
            self.log("[live] stopped")

    def start_job(self, operation, name):
        with self.worker_lock:
            if self.worker is not None and self.worker.is_alive():
                raise RuntimeError("another job is running: {}".format(self.worker_name))
            self.stop_event.clear()
            self.worker_name = name

            def runner():
                try:
                    operation()
                except Exception:
                    self.base.stop()
                    self.last_result = "error"
                    self.log(traceback.format_exc())
                finally:
                    self.worker_name = None

            self.worker = threading.Thread(target=runner, name=name)
            self.worker.daemon = True
            self.worker.start()
        return name

    def stop_base(self):
        self.stop_event.set()
        self.base.stop()
        self.log("[stop] base stop requested")

    def close(self):
        self.stop_base()
        worker = self.worker
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(3.0)
        self.stop_media()
        if worker is not None and worker.is_alive():
            self.log("[camera] worker still stopping; release deferred")
            return False
        self.depth.stop()
        self.camera_started = False
        self.log("[camera] released")
        return True


def build_widget():
    import ipywidgets as widgets
    from IPython.display import FileLink, clear_output, display

    defaults = load_config()
    default_stop = defaults.get("navigation.can.approach.stop")
    output = widgets.Output(layout=widgets.Layout(border="1px solid #bbb", height="220px", overflow_y="auto"))
    image = widgets.Image(format="jpeg", layout=widgets.Layout(width="640px"))
    base_real = widgets.Checkbox(value=False, description="REAL base")
    camera_real = widgets.Checkbox(value=True, description="REAL camera")
    live = widgets.Checkbox(value=True, description="Live HUD")
    record = widgets.Checkbox(value=False, description="Record")
    mode = widgets.Dropdown(options=list(APPROACH_STOP_MODES), value=default_stop["mode"], description="stop mode")
    bbox_threshold = widgets.FloatText(value=float(default_stop.get("bbox_height_threshold", 0.36)), description="bbox height")
    depth_threshold = widgets.FloatText(value=float(default_stop.get("depth_raw_threshold", 2.05)), description="depth raw")
    speed = widgets.FloatText(value=float(defaults.get("navigation.can.approach.speed")), description="speed")
    timeout = widgets.FloatText(value=float(defaults.get("navigation.can.approach.timeout_seconds")), description="timeout s")
    fps = widgets.FloatText(value=2.0, description="control fps")
    ideal_cm = widgets.FloatText(value=10.0, description="ideal stop cm")
    measured_cm = widgets.FloatText(value=10.0, description="measured cm")
    state = {"session": None}

    def log(message):
        with output:
            print(message)

    def make_session():
        old = state.get("session")
        if old is not None:
            old.close()
        session = CanStableStopSession(
            base_real=base_real.value, camera_real=camera_real.value,
            mode=mode.value, bbox_height_threshold=bbox_threshold.value,
            depth_raw_threshold=depth_threshold.value, speed=speed.value,
            timeout_seconds=timeout.value, control_fps=fps.value,
            ideal_stop_distance_cm=ideal_cm.value, live=live.value, record=record.value,
        )
        session.log_callback = log
        session.frame_callback = lambda value: setattr(image, "value", value)
        state["session"] = session
        log("[session] ready output={}".format(session.output_directory))
        return session

    def current():
        return state.get("session") or make_session()

    def guarded(action):
        def callback(_button):
            try:
                action()
            except Exception:
                log(traceback.format_exc())
        return callback

    buttons = {
        "new": widgets.Button(description="Apply UI / New Session", button_style="info"),
        "load": widgets.Button(description="Load Camera + Models"),
        "sample": widgets.Button(description="Sample Once"),
        "live": widgets.Button(description="Start Live"),
        "trial": widgets.Button(description="Run Stop Trial", button_style="success"),
        "stop": widgets.Button(description="STOP BASE", button_style="danger"),
        "media": widgets.Button(description="Stop Media"),
        "measure": widgets.Button(description="Save Measurement"),
        "link": widgets.Button(description="Recording / CSV Link"),
        "release": widgets.Button(description="Release Camera"),
        "clear": widgets.Button(description="Clear Log"),
    }
    buttons["new"].on_click(guarded(make_session))
    buttons["load"].on_click(guarded(lambda: current().start_camera()))
    buttons["sample"].on_click(guarded(lambda: log("[sample] {}".format(current().sample()))))
    buttons["live"].on_click(guarded(lambda: current().start_job(current().run_live, "live")))
    buttons["trial"].on_click(guarded(lambda: current().start_job(current().run_trial, "trial")))
    buttons["stop"].on_click(guarded(lambda: current().stop_base()))
    buttons["media"].on_click(guarded(lambda: current().stop_media()))
    buttons["measure"].on_click(guarded(lambda: current().record_measurement(measured_cm.value)))

    def show_links():
        session = current()
        with output:
            display(FileLink(os.path.relpath(session.csv_path, os.getcwd()), result_html_prefix="CSV: "))
            if session.recording_path and os.path.isfile(session.recording_path):
                display(FileLink(os.path.relpath(session.recording_path, os.getcwd()), result_html_prefix="Video: "))

    buttons["link"].on_click(guarded(show_links))
    buttons["release"].on_click(guarded(lambda: current().close()))

    def clear_log():
        with output:
            clear_output(wait=False)

    buttons["clear"].on_click(guarded(clear_log))
    panel = widgets.VBox([
        widgets.HTML("<b>Can stable pre-arm-down stop tuning</b> — manually align the can, set the expected final gap, then run one longitudinal trial."),
        widgets.HBox([base_real, camera_real, live, record]),
        widgets.HBox([mode, bbox_threshold, depth_threshold]),
        widgets.HBox([speed, timeout, fps]),
        widgets.HBox([ideal_cm, measured_cm, buttons["measure"]]),
        widgets.HBox([buttons["new"], buttons["load"], buttons["sample"], buttons["live"]]),
        widgets.HBox([buttons["trial"], buttons["stop"], buttons["media"], buttons["release"]]),
        widgets.HBox([buttons["link"], buttons["clear"]]),
        image,
        output,
    ])
    display(panel)
    return state


__all__ = ("CanStableStopSession", "build_widget")
