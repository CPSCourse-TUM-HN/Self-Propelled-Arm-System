from __future__ import print_function

import json
import math
import os
import sys
import threading
import time
import traceback

import cv2
import ipywidgets as widgets
from IPython.display import FileLink, display

from demo_core import (
    DemoStateMachine,
    HudEventRecorder,
    MissionState,
    TargetType,
    load_config,
    render_hud,
    runtime_hud_snapshot,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def critical_parameter_snapshot(config):
    """Return only the JSON values needed to interpret a docking test."""
    exp2 = config.get("navigation.bin.side_docking.experimental")
    return {
        "calibration_yaml": config.get("camera.calibration_yaml"),
        "marker_length_m": config.get("detectors.bin.marker_length_m"),
        "front_align_tolerance_norm": config.get("navigation.bin.align.tolerance_norm"),
        "front_align_target_error_x_norm": config.get(
            "navigation.bin.align.target_error_x_norm", 0.0
        ),
        "front_approach_speed": config.get("navigation.bin.approach.speed"),
        "front_approach_stop": config.get("navigation.bin.approach.stop"),
        "front_pnp_z_stop_threshold_m": config.get(
            "navigation.bin.approach.stop.pnp_z_threshold_m"
        ),
        "side_correction_mode": exp2.get("correction_mode"),
        "center_tolerance_norm": exp2.get("center_tolerance_norm"),
        "yaw_tolerance_deg": math.degrees(float(exp2.get("yaw_tolerance_rad", 0.0))),
        "target_yaw_deg": math.degrees(float(exp2.get("target_yaw_rad", 0.0))),
        "center_stage_entry_yaw_deg": math.degrees(
            float(exp2.get("center_stage_entry_yaw_rad", 0.0))
        ),
        "coarse_yaw_center_guard_norm": exp2.get("coarse_yaw_center_guard_norm"),
        "center_correction_seconds": exp2.get("center_correction_seconds"),
        "yaw_correction_seconds": exp2.get("yaw_correction_seconds"),
        "yaw_single_correction_max_seconds": exp2.get("yaw_single_correction_max_seconds"),
        "stable_frames": exp2.get("stable_frames"),
    }


class ThreadOutputRouter(object):
    """Route only the worker thread to disk, leaving notebook output open."""

    def __init__(self, fallback):
        self.fallback = fallback
        self.routes = {}
        self.lock = threading.Lock()

    def bind(self, stream):
        with self.lock:
            self.routes[threading.get_ident()] = stream

    def unbind(self):
        with self.lock:
            self.routes.pop(threading.get_ident(), None)

    def _target(self):
        with self.lock:
            return self.routes.get(threading.get_ident(), self.fallback)

    def write(self, data):
        return self._target().write(data)

    def flush(self):
        return self._target().flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.fallback, name)


STDOUT_ROUTER = ThreadOutputRouter(sys.stdout)
STDERR_ROUTER = ThreadOutputRouter(sys.stderr)
sys.stdout = STDOUT_ROUTER
sys.stderr = STDERR_ROUTER


class BinDockingTuningPanel(object):
    """One-click bin search/alignment through optional insertion and release."""

    media_fps = 4.0

    def __init__(self):
        self.camera_real = widgets.Checkbox(value=True, description="camera_real")
        self.base_real = widgets.Checkbox(value=True, description="base_real")
        self.arm_real = widgets.Checkbox(value=True, description="arm_real")
        self.live = widgets.Checkbox(value=True, description="live")
        self.record = widgets.Checkbox(value=False, description="record")
        self.start_stage = widgets.Dropdown(
            options=(("Search", "search"), ("Align", "align")),
            value="align",
            description="start_stage",
        )
        self.release_after_correction = widgets.Checkbox(
            value=False,
            description="release_after_correction",
        )
        self.image = widgets.Image(format="jpeg", width=640, height=480)
        self.status = widgets.HTML(value="<b>idle</b>")
        self.output = widgets.Output(
            layout={"border": "1px solid #bbb", "height": "170px", "overflow_y": "auto"}
        )
        self.runtime = None
        self.worker_thread = None
        self.media_thread = None
        self.media_running = False
        self.last_recording_path = None
        self.last_events_path = None
        self.session_dir = os.path.join(
            PROJECT_ROOT, "logs", "bin_docking_tuning", time.strftime("%Y%m%d_%H%M%S")
        )
        if not os.path.isdir(self.session_dir):
            os.makedirs(self.session_dir)
        self._buttons = self._build_buttons()
        self.reload_json()

    def _log(self, text, error=False):
        append = self.output.append_stderr if error else self.output.append_stdout
        append(str(text).rstrip() + "\n")

    def _config_overrides(self):
        return {
            "runtime": {
                "dry_run": {
                    "camera": not bool(self.camera_real.value),
                    "base": not bool(self.base_real.value),
                    "arm": not bool(self.arm_real.value),
                },
                "max_pickups": 1,
            },
            "navigation": {
                "bin": {"side_docking": {"experimental": {"enabled": True}}},
            },
        }

    def _worker_alive(self):
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def reload_json(self):
        if self._worker_alive():
            raise RuntimeError("docking run is active; STOP ALL before reloading JSON")
        self.stop_media()
        if self.runtime is not None:
            self.runtime.services.base.stop()
            self.runtime.release_camera()
        config = load_config(overrides=self._config_overrides())
        self.runtime = DemoStateMachine(config)
        self._log("[json] reloaded {}".format(config.parameters_path))
        return critical_parameter_snapshot(config)

    def _prepare(self):
        parameters = self.reload_json()
        rt = self.runtime
        rt.navigator.frame_observer = None
        rt.stop_requested = False
        rt.pause_requested = False
        rt.exp2_side_docked = False
        rt.context.grabbed = True
        rt.context.target_type = TargetType.BIN
        rt.services.base.stop()

        targets = rt.services.arm.pose("carry")
        if not rt.services.arm.wait_for_positions(targets, "carry"):
            raise RuntimeError("carry pose did not settle")
        rt.services.depth.start(camera_only=True)
        if rt.config.get("runtime.dry_run.camera", True):
            rt.services.bin_detector._load_calibration()
        else:
            rt.services.bin_detector.load()
        if self.start_stage.value == "align":
            frame = rt.services.depth.read_frame()
            if frame is None:
                raise RuntimeError("camera returned no frame")
            observation = rt.services.bin_detector.detect(frame)
            if not observation.get("found"):
                raise RuntimeError("Tag is not visible; place it in view or start from search")
            rt.context.remember_target(TargetType.BIN, observation)
            rt.previous_state = MissionState.SEARCHING
            rt.state = MissionState.ALIGNING
        else:
            rt.previous_state = MissionState.PLANNING
            rt.state = MissionState.SEARCHING
        rt.context.begin_state(rt.state)
        return rt, parameters

    @staticmethod
    def _number(value, digits=1):
        try:
            return ("{:.%df}" % digits).format(float(value))
        except (TypeError, ValueError):
            return "n/a"

    def _update_status(self, snapshot):
        self.status.value = (
            "<b>{}</b> &nbsp; {} &nbsp; cx={} &nbsp; pnp_z={}/{}m &nbsp; "
            "raw={}deg &nbsp; ctrl={}deg &nbsp; target={}deg"
        ).format(
            snapshot.get("state", "?"),
            snapshot.get("subphase", "?"),
            self._number(snapshot.get("error_x"), 3),
            self._number(snapshot.get("pose_z"), 3),
            self._number(self.runtime.config.get(
                "navigation.bin.approach.stop.pnp_z_threshold_m"
            ), 3),
            self._number(snapshot.get("raw_yaw_deg"), 1),
            self._number(snapshot.get("yaw_control_error_deg"), 1),
            self._number(snapshot.get("yaw_target_deg"), 1),
        )

    def _media_canvas(self, rt, recording):
        frame = rt.services.depth.read_frame()
        if frame is None:
            return None, None
        snapshot = runtime_hud_snapshot(
            rt, depth_stats=None, frame_available=True, recording=recording
        )
        canvas = render_hud(frame, snapshot, rt.config.section("camera"), enabled=True)
        return canvas, snapshot

    def _media_worker(self, rt):
        writer = None
        recorder = None
        started_at = None
        try:
            while self.media_running:
                recording = bool(self.record.value)
                canvas, snapshot = self._media_canvas(rt, recording)
                if canvas is not None:
                    self._update_status(snapshot)
                    if self.live.value:
                        ok, encoded = cv2.imencode(
                            ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 76]
                        )
                        if ok:
                            self.image.value = encoded.tobytes()
                    if recording and writer is None:
                        stamp = time.strftime("%H%M%S")
                        self.last_recording_path = os.path.join(
                            self.session_dir, "docking_{}.avi".format(stamp)
                        )
                        self.last_events_path = os.path.join(
                            self.session_dir, "docking_{}.events.csv".format(stamp)
                        )
                        writer = cv2.VideoWriter(
                            self.last_recording_path,
                            cv2.VideoWriter_fourcc(*"MJPG"),
                            self.media_fps,
                            (640, 480),
                        )
                        if not writer.isOpened():
                            writer.release()
                            writer = None
                            raise RuntimeError("recording writer could not open")
                        recorder = HudEventRecorder(self.last_events_path)
                        started_at = time.time()
                        self._log("[record] started {}".format(self.last_recording_path))
                    if writer is not None and recording:
                        writer.write(canvas)
                        recorder.write(snapshot, time.time() - started_at)
                    elif writer is not None:
                        writer.release()
                        recorder.close()
                        writer = recorder = None
                        self._log("[record] stopped")
                time.sleep(1.0 / self.media_fps)
        except Exception as exc:
            self._log("[media] {}".format(exc), error=True)
        finally:
            if writer is not None:
                writer.release()
            if recorder is not None:
                recorder.close()
            self.media_running = False
            self.media_thread = None

    def start_media(self):
        if not self.live.value and not self.record.value:
            return {"started": False, "reason": "live and record are both disabled"}
        if self.media_thread is not None and self.media_thread.is_alive():
            return {"started": False, "reason": "media already running"}
        canvas, snapshot = self._media_canvas(self.runtime, bool(self.record.value))
        if canvas is None:
            raise RuntimeError("camera returned no media frame")
        self._update_status(snapshot)
        if self.live.value:
            ok, encoded = cv2.imencode(".jpg", canvas)
            if ok:
                self.image.value = encoded.tobytes()
        self.media_running = True
        self.media_thread = threading.Thread(target=self._media_worker, args=(self.runtime,))
        self.media_thread.daemon = True
        self.media_thread.start()
        return {"started": True, "live": bool(self.live.value), "record": bool(self.record.value)}

    def stop_media(self, wait_seconds=1.5):
        self.media_running = False
        thread = self.media_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(float(wait_seconds))
        alive = thread is not None and thread.is_alive()
        if not alive:
            self.media_thread = None
        return {"stopped": not alive}

    def _run_worker(self, rt, log_path):
        result = "FAILED"
        previous = rt.state
        with open(log_path, "a") as log_file:
            STDOUT_ROUTER.bind(log_file)
            STDERR_ROUTER.bind(log_file)
            try:
                print("[docking-test] parameters={}".format(
                    json.dumps(critical_parameter_snapshot(rt.config), sort_keys=True)
                ))
                while not rt.stop_requested:
                    rt.interrupt_point()
                    previous = rt.state
                    outcome = rt.step_once()
                    if rt.state != previous:
                        transition = rt.last_transition or {}
                        self._log(
                            "[state] {} -> {} ({})".format(
                                previous.value,
                                rt.state.value,
                                transition.get("event", "?"),
                            )
                        )
                    completion = self.completion_result(previous, rt.state)
                    if completion == "CORRECTION_COMPLETE":
                        result = completion
                        print("[docking-test] stopped before release")
                        break
                    if completion == "RELEASE_COMPLETE":
                        result = completion
                        print("[docking-test] insert, release, and safe-home complete")
                        break
                    if rt.state in (MissionState.INTERMEDIATE, MissionState.FAILED, MissionState.DONE):
                        result = "FAILED_{}".format(rt.state.value)
                        break
                    time.sleep(float(rt.config.get("runtime.loop_pause_seconds", 0.02)))
                if rt.stop_requested:
                    result = "STOPPED"
            except Exception:
                if rt.stop_requested:
                    result = "STOPPED"
                else:
                    traceback.print_exc()
                    self._log("[run] failed; details are in {}".format(log_path), error=True)
            finally:
                rt.services.base.stop()
                STDERR_ROUTER.unbind()
                STDOUT_ROUTER.unbind()
        self.stop_media()
        self.worker_thread = None
        self._log("[run] {} log={}".format(result, log_path))

    def completion_result(self, previous_state, current_state):
        if current_state == MissionState.FINALIZING and not bool(self.release_after_correction.value):
            return "CORRECTION_COMPLETE"
        if (
            bool(self.release_after_correction.value)
            and previous_state == MissionState.FINALIZING
            and current_state == MissionState.PLANNING
        ):
            return "RELEASE_COMPLETE"
        return None

    def run_docking(self):
        if self._worker_alive():
            return {"started": False, "reason": "docking run already active"}
        rt, parameters = self._prepare()
        self._log("[parameters] {}".format(json.dumps(parameters, sort_keys=True)))
        if self.live.value or self.record.value:
            self.start_media()
        log_path = os.path.join(
            self.session_dir, "docking_run_{}.log".format(time.strftime("%H%M%S"))
        )
        self.worker_thread = threading.Thread(target=self._run_worker, args=(rt, log_path))
        self.worker_thread.daemon = True
        self.worker_thread.start()
        return {
            "started": True,
            "from": self.start_stage.value,
            "until": "release" if self.release_after_correction.value else "correction",
            "release_executed": bool(self.release_after_correction.value),
            "log": log_path,
        }

    def stop_all(self):
        rt = self.runtime
        if rt is None:
            return {"stopped": True}
        rt.request_stop()
        rt.services.base.stop()
        positions = rt.services.arm.stop_and_hold()
        self.stop_media()
        rt.release_camera()
        return {"stopped": True, "arm_hold": positions}

    def recording_link(self):
        if not self.last_recording_path or not os.path.isfile(self.last_recording_path):
            return {"available": False, "reason": "no recording in this kernel"}
        display(FileLink(self.last_recording_path))
        display(FileLink(self.last_events_path))
        return {
            "available": True,
            "video": self.last_recording_path,
            "events": self.last_events_path,
        }

    def clear_log(self):
        self.output.clear_output()
        return True

    def _callback(self, fn):
        def wrapped(_=None):
            try:
                result = fn()
                self._log("[result] {}".format(result))
            except Exception as exc:
                self._log("[error] {}".format(exc), error=True)
        return wrapped

    def _build_buttons(self):
        specs = (
            ("Reload JSON", self.reload_json, ""),
            ("RUN DOCKING", self.run_docking, "success"),
            ("STOP ALL", self.stop_all, "danger"),
            ("Recording Link", self.recording_link, ""),
            ("Clear Log", self.clear_log, ""),
        )
        buttons = []
        for label, fn, style in specs:
            button = widgets.Button(
                description=label, button_style=style, layout=widgets.Layout(width="155px")
            )
            button.on_click(self._callback(fn))
            buttons.append(button)
        return buttons

    def widget(self):
        return widgets.VBox([
            widgets.HBox([self.camera_real, self.base_real, self.arm_real]),
            widgets.HBox([self.start_stage, self.release_after_correction]),
            widgets.HBox([self.live, self.record]),
            widgets.HBox(self._buttons),
            self.status,
            self.image,
            self.output,
        ])
