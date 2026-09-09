"""Run the motion-safe pre-demo hardware and dry-run pipeline smoke check."""

from __future__ import print_function

import argparse
import contextlib
import gc
import io
import json
import os
import sys
import time
import traceback


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from demo_core import DemoDiagnostics, DemoStateMachine, MissionState, load_config
from demo_core.logging_utils import TeeLogger, default_log_path


class SmokeReport(object):
    def __init__(self):
        self.entries = []

    def add(self, level, phase, message, details=None):
        entry = {
            "level": str(level),
            "phase": str(phase),
            "message": str(message),
        }
        if details is not None:
            entry["details"] = details
        self.entries.append(entry)
        suffix = " details={}".format(details) if details is not None else ""
        print("[smoke][{}][{}] {}{}".format(level, phase, message, suffix))

    def passed(self, phase, message, details=None):
        self.add("PASS", phase, message, details)

    def warning(self, phase, message, details=None):
        self.add("WARN", phase, message, details)

    def failed(self, phase, message, details=None):
        self.add("FAIL", phase, message, details)

    def has_failures(self):
        return any(entry["level"] == "FAIL" for entry in self.entries)

    def summary(self):
        counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
        for entry in self.entries:
            counts[entry["level"]] += 1
        return {"counts": counts, "entries": self.entries}


def check_assets(config, report):
    phase = "assets"
    required = [
        ("config", config.config_path, 1),
        ("parameters", config.parameters_path, 1),
        ("can_model", config.resolve_path(config.get("detectors.can.model_path")), 1024 * 1024),
        ("can_labels", config.resolve_path(config.get("detectors.can.labels_path")), 1),
        ("apriltag_image", config.resolve_path("assets/bin_apriltag_36h11_id_0.png"), 1),
    ]
    for name, path, minimum_size in required:
        if not path or not os.path.isfile(path):
            report.failed(phase, "{} missing".format(name), path)
            continue
        size = os.path.getsize(path)
        if size < minimum_size:
            report.failed(phase, "{} is unexpectedly small".format(name), {"path": path, "bytes": size})
        else:
            report.passed(phase, "{} available".format(name), {"bytes": size})


def check_board_imports(report):
    phase = "imports"
    checks = (
        ("cv2", lambda: __import__("cv2")),
        ("jetbot", lambda: __import__("jetbot")),
        ("jetson_inference", lambda: __import__("jetson_inference")),
        ("jetson_utils", lambda: __import__("jetson_utils")),
        ("pupil_apriltags", lambda: __import__("pupil_apriltags")),
    )
    for name, operation in checks:
        try:
            module = operation()
            report.passed(phase, "{} import ok".format(name), getattr(module, "__version__", None))
        except Exception as exc:
            report.failed(phase, "{} import failed".format(name), str(exc))


def summarize_observation(observation):
    if not observation:
        return None
    keys = (
        "kind",
        "found",
        "confidence",
        "id",
        "bbox",
        "center_x",
        "center_y",
        "error_x",
        "bbox_height_norm",
        "edge_length_norm",
    )
    return {key: observation.get(key) for key in keys if key in observation}


def probe_camera_and_models(config, report, frame_count):
    phase = "camera_models"
    diagnostics = DemoDiagnostics(config)
    try:
        frame = diagnostics.start_camera()
        if frame is None:
            report.failed(phase, "camera started but returned no frame")
            return
        report.passed(phase, "camera-only mode started", {"shape": list(frame.shape), "dtype": str(frame.dtype)})

        frames_ok = 0
        for index in range(max(1, int(frame_count))):
            frame = diagnostics.read_frame()
            if frame is not None:
                frames_ok += 1
                print("[smoke][frame] {}/{} shape={}".format(index + 1, frame_count, frame.shape))
            time.sleep(0.1)
        if frames_ok == max(1, int(frame_count)):
            report.passed(phase, "camera frame continuity ok", {"frames": frames_ok})
        else:
            report.failed(phase, "camera frame continuity failed", {"expected": frame_count, "received": frames_ok})

        depth = diagnostics.observe_depth()
        lens = depth.get("lens") if depth else None
        if lens and int(lens.get("count", 1)) > 0:
            report.passed(phase, "DepthNet inference ok", {
                "mean": lens.get("mean"),
                "min": lens.get("min"),
                "max": lens.get("max"),
            })
        else:
            report.failed(phase, "DepthNet returned no usable center ROI", depth)

        diagnostics.load_can(reload_model=False)
        if diagnostics.services.can_detector.net is None:
            report.failed(phase, "can detectNet did not create a network")
        else:
            report.passed(phase, "can detectNet loaded")
        can = diagnostics.observe_can()
        if can and can.get("found"):
            report.passed(phase, "can inference call found a target", summarize_observation(can))
        else:
            report.warning(phase, "can inference call succeeded but no can was visible", summarize_observation(can))

        diagnostics.load_tag(reload_model=False)
        tag_service = diagnostics.services.bin_detector
        if tag_service.detector is None and tag_service.aruco_dict is None:
            report.failed(phase, "AprilTag detector did not initialize")
        else:
            report.passed(phase, "AprilTag detector loaded", {"backend": tag_service.backend})
        tag = diagnostics.observe_tag()
        if tag and tag.get("found"):
            report.passed(phase, "AprilTag inference call found a target", summarize_observation(tag))
        else:
            report.warning(phase, "AprilTag inference call succeeded but no tag was visible", summarize_observation(tag))
    except Exception as exc:
        report.failed(phase, "camera/model probe raised an exception", str(exc))
        traceback.print_exc()
    finally:
        try:
            diagnostics.release_camera()
        except Exception as exc:
            report.failed("cleanup", "camera release failed", str(exc))
        try:
            diagnostics.reset_models()
        except Exception as exc:
            report.warning("cleanup", "model reset raised an exception", str(exc))
        del diagnostics
        gc.collect()
        time.sleep(1.0)
        report.passed("cleanup", "real camera/model handles released")


def run_simulated_mission(report, max_ticks):
    phase = "simulated_mission"
    config = load_config(overrides={
        "runtime": {
            "dry_run": {"camera": True, "base": True, "arm": True},
            "loop_pause_seconds": 0.0,
        },
    })
    runtime = DemoStateMachine(config)
    try:
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            success = runtime.run(max_ticks=int(max_ticks))
        for line in captured.getvalue().splitlines():
            if (
                line.startswith("[fsm] ") and "--" in line
                or line.startswith("[mission]")
                or line.startswith("[map] pose reset")
                or line.startswith("[map] patrol index=") and "complete=True" in line
                or "unhandled error" in line
                or "Traceback" in line
            ):
                print("[smoke][flow] {}".format(line))
        snapshot = runtime.context.snapshot()
        map_snapshot = snapshot.get("vague_map") or {}
        details = {
            "terminal_state": runtime.state.value,
            "completed_pickups": snapshot.get("completed_pickups"),
            "known_cans": len(map_snapshot.get("known_cans", [])),
        }
        if success and runtime.state == MissionState.DONE:
            report.passed(phase, "full dry-run FSM reached DONE", details)
        else:
            report.failed(phase, "full dry-run FSM did not reach DONE", details)
    except Exception as exc:
        report.failed(phase, "simulated mission raised an exception", str(exc))
        traceback.print_exc()


def execute(args):
    report = SmokeReport()
    print("[smoke] HOME PIPELINE SMOKE TEST")
    print("[smoke] SAFETY: base dry-run=True, arm dry-run=True; no motion commands reach hardware")
    print("[smoke] camera_real={}".format(not args.simulation_only))

    try:
        config = load_config(overrides={
            "runtime": {"dry_run": {"camera": bool(args.simulation_only), "base": True, "arm": True}},
        })
        report.passed("config", "configuration merged and validated")
    except Exception as exc:
        report.failed("config", "configuration failed validation", str(exc))
        print(json.dumps(report.summary(), indent=2, sort_keys=True))
        return 1

    check_assets(config, report)
    if not args.simulation_only:
        check_board_imports(report)
        probe_camera_and_models(config, report, args.frames)
    else:
        report.warning("camera_models", "real camera/model probe skipped by --simulation-only")

    run_simulated_mission(report, args.max_ticks)
    summary = report.summary()
    print("\n[smoke] FINAL SUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if report.has_failures():
        print("[smoke] RESULT=FAIL; inspect FAIL entries before the demo")
        return 1
    print("[smoke] RESULT=PASS_WITH_WARNINGS" if summary["counts"]["WARN"] else "[smoke] RESULT=PASS")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Short safe home diagnostic: one real camera/model probe plus one fully simulated mission."
    )
    parser.add_argument("--simulation-only", action="store_true", help="skip Jetson camera/model probe")
    parser.add_argument("--frames", type=int, default=2, help="camera continuity frames; default 2")
    parser.add_argument("--max-ticks", type=int, default=3000, help="dry-run FSM safety limit")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--no-log-file", action="store_true")
    args = parser.parse_args()
    if args.frames <= 0 or args.max_ticks <= 0:
        parser.error("--frames and --max-ticks must be positive")
    if args.no_log_file:
        return execute(args)
    path = args.log_file or default_log_path("home_pipeline_smoke")
    with TeeLogger(path):
        result = execute(args)
    print("[smoke] log={}".format(path))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
