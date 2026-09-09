from __future__ import print_function

import argparse
import csv
import json
import os
import sys
import threading
import time
import uuid


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_PARAMETERS = os.path.join(SCRIPT_DIR, "base_speed_time_parameters.json")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from demo_core import load_config
from demo_core.robot_control import BaseController


def load_parameters(path):
    with open(path, "r") as stream:
        data = json.load(stream)
    cases = data.get("cases", [])
    if not cases:
        raise ValueError("parameters file contains no cases")
    names = set()
    for case in cases:
        name = str(case.get("name", "")).strip()
        motion = str(case.get("motion", "")).lower()
        speed = float(case.get("speed", 0.0))
        seconds = float(case.get("seconds", 0.0))
        repeat = int(case.get("repeat", 1))
        if not name or name in names:
            raise ValueError("case names must be non-empty and unique: {}".format(name))
        if motion not in ("forward", "left", "right"):
            raise ValueError("{} has unsupported motion {}".format(name, motion))
        if speed <= 0.0 or speed > float(data.get("maximum_speed", 0.8)):
            raise ValueError("{} speed is outside the configured safety range".format(name))
        if seconds <= 0.0 or seconds > float(data.get("maximum_seconds", 5.0)):
            raise ValueError("{} seconds is outside the configured safety range".format(name))
        if repeat <= 0:
            raise ValueError("{} repeat must be positive".format(name))
        names.add(name)
    return data


def select_cases(data, group, case_name):
    cases = list(data["cases"])
    if case_name:
        selected = [case for case in cases if case["name"] == case_name]
        if not selected:
            raise ValueError("unknown case: {}".format(case_name))
        return selected
    if group == "forward":
        return [case for case in cases if case["motion"] == "forward"]
    if group == "turn":
        return [case for case in cases if case["motion"] in ("left", "right")]
    return cases


def output_path(config):
    root = config.resolve_path(config.get("paths.logs", "logs"))
    directory = os.path.join(root, "base_speed_time")
    if not os.path.isdir(directory):
        os.makedirs(directory)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(directory, "base_speed_time_{}.csv".format(stamp))


def prompt_measurement(case):
    if case["motion"] == "forward":
        prompt = "Measured forward distance in meters (blank to skip): "
        kind = "distance_m"
    else:
        prompt = "Measured absolute turn angle in degrees (blank to skip): "
        kind = "angle_deg"
    value = input(prompt).strip()
    return kind, float(value) if value else ""


def turn_response_metrics(measured_angle_deg, speed, seconds):
    measured_angle_deg = abs(float(measured_angle_deg))
    speed = float(speed)
    seconds = float(seconds)
    if measured_angle_deg <= 0.0 or speed <= 0.0 or seconds <= 0.0:
        raise ValueError("turn measurement, speed, and seconds must be positive")
    return {
        "turn_deg_per_speed_second": measured_angle_deg / (speed * seconds),
        "recommended_90_seconds": 90.0 * seconds / measured_angle_deg,
    }


def annotate_measurement(path, run_id, measurement_value):
    with open(path, "r", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("calibration CSV has no rows")
    fieldnames = list(rows[0].keys())
    matched = False
    for row in rows:
        if row.get("run_id") != run_id:
            continue
        matched = True
        value = float(measurement_value)
        row["measurement_kind"] = "angle_deg" if row.get("motion") in ("left", "right") else "distance_m"
        row["measurement_value"] = str(value)
        if row.get("motion") in ("left", "right"):
            metrics = turn_response_metrics(value, row["effective_speed"], row["seconds"])
            row["turn_deg_per_speed_second"] = "{:.9f}".format(metrics["turn_deg_per_speed_second"])
            row["recommended_90_seconds"] = "{:.9f}".format(metrics["recommended_90_seconds"])
    if not matched:
        raise ValueError("run_id not found: {}".format(run_id))

    means = {}
    for direction in ("left", "right"):
        values = [
            float(row["turn_deg_per_speed_second"])
            for row in rows
            if row.get("motion") == direction and row.get("turn_deg_per_speed_second")
        ]
        if values:
            means[direction] = sum(values) / float(len(values))
    if len(means) == 2:
        target = (means["left"] + means["right"]) / 2.0
        for row in rows:
            direction = row.get("motion")
            if direction in means:
                row["suggested_direction_scale"] = "{:.9f}".format(target / means[direction])

    temporary = path + ".tmp"
    with open(temporary, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return next(row for row in rows if row.get("run_id") == run_id)


def _run_interruptible_motion(controller, motion, speed, seconds, label, stop_event):
    started_at = time.time()
    controller.start_motion(motion, speed, label)
    try:
        while time.time() - started_at < seconds:
            if stop_event.is_set():
                return False
            remaining = seconds - (time.time() - started_at)
            time.sleep(min(0.05, max(0.0, remaining)))
        return True
    finally:
        controller.stop()


def run_cases(data, selected, real_motion, controller=None, stop_event=None, surface_label="unspecified"):
    dry_run = not bool(real_motion)
    config = load_config(overrides={"runtime": {"dry_run": {"base": dry_run}}})
    controller = controller or BaseController(config)
    stop_event = stop_event or threading.Event()
    path = output_path(config)
    settle = max(0.0, float(data.get("settle_seconds_between_runs", 1.0)))
    prompt = bool(data.get("prompt_for_measurement", False))
    fieldnames = [
        "run_id", "timestamp", "surface_label", "case", "motion", "requested_speed",
        "command_scale", "effective_speed", "seconds", "repeat_index", "dry_run",
        "measurement_kind", "measurement_value", "turn_deg_per_speed_second",
        "recommended_90_seconds", "suggested_direction_scale", "result",
    ]
    run_ids = []
    print("[calibration] mode={} output={}".format("REAL" if real_motion else "DRY_RUN", path))
    try:
        with open(path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for case in selected:
                for repeat_index in range(1, int(case.get("repeat", 1)) + 1):
                    if stop_event.is_set():
                        print("[calibration] stop requested before next case")
                        return {"path": path, "run_ids": run_ids}
                    motion = str(case["motion"])
                    speed = float(case["speed"])
                    command_scale = float(config.get("base_motion_command_scales.{}".format(motion), 1.0))
                    effective_speed = speed * command_scale
                    seconds = float(case["seconds"])
                    run_id = uuid.uuid4().hex[:12]
                    run_ids.append(run_id)
                    print(
                        "[calibration] case={} repeat={} motion={} speed={} seconds={}".format(
                            case["name"], repeat_index, motion, speed, seconds
                        )
                    )
                    measurement_kind = ""
                    measurement_value = ""
                    result = "ok"
                    try:
                        completed = _run_interruptible_motion(
                            controller,
                            motion,
                            speed,
                            seconds,
                            "calibration_{}".format(case["name"]),
                            stop_event,
                        )
                        if not completed:
                            result = "stopped"
                        elif real_motion and prompt:
                            measurement_kind, measurement_value = prompt_measurement(case)
                    except Exception:
                        result = "failed"
                        raise
                    finally:
                        controller.stop()
                        writer.writerow({
                            "run_id": run_id,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "surface_label": str(surface_label).strip() or "unspecified",
                            "case": case["name"],
                            "motion": motion,
                            "requested_speed": speed,
                            "command_scale": command_scale,
                            "effective_speed": effective_speed,
                            "seconds": seconds,
                            "repeat_index": repeat_index,
                            "dry_run": dry_run,
                            "measurement_kind": measurement_kind,
                            "measurement_value": measurement_value,
                            "turn_deg_per_speed_second": "",
                            "recommended_90_seconds": "",
                            "suggested_direction_scale": "",
                            "result": result,
                        })
                        stream.flush()
                    if settle > 0.0:
                        if stop_event.wait(settle):
                            print("[calibration] stopped during settle interval")
                            return {"path": path, "run_ids": run_ids}
    finally:
        controller.stop()
    print("[calibration] complete output={}".format(path))
    return {"path": path, "run_ids": run_ids}


def build_ui(parameters_path=DEFAULT_PARAMETERS):
    """Build the Notebook-only wrapper without adding ipywidgets to CLI startup."""
    import ipywidgets as widgets
    from IPython.display import display

    state = {
        "data": None,
        "thread": None,
        "controller": None,
        "stop_event": threading.Event(),
        "last_output": None,
        "last_run_id": None,
    }
    output = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #b8b8b8",
            height="320px",
            overflow_y="auto",
            width="100%",
        )
    )
    case_name = widgets.Dropdown(description="Case", options=[])
    surface_label = widgets.Text(value="surface_a", description="Surface")
    measured_value = widgets.FloatText(value=90.0, description="Measured")
    real_motion = widgets.Checkbox(value=False, description="Real base motion")
    confirm_multiple = widgets.Checkbox(value=False, description="Allow multiple real cases")
    reload_button = widgets.Button(description="Reload JSON", button_style="info")
    list_button = widgets.Button(description="Print Cases")
    run_case_button = widgets.Button(description="Run Selected", button_style="success")
    run_forward_button = widgets.Button(description="Run Forward Group")
    run_turn_button = widgets.Button(description="Run Turn Group")
    stop_button = widgets.Button(description="STOP BASE", button_style="danger")
    record_button = widgets.Button(description="Record Measurement", button_style="warning")
    clear_button = widgets.Button(description="Clear Log")

    def log(message):
        output.append_stdout(str(message) + "\n")

    def reload_parameters(_=None):
        try:
            data = load_parameters(parameters_path)
            state["data"] = data
            case_name.options = [(case["name"], case["name"]) for case in data["cases"]]
            log("[parameters] reloaded {} cases from {}".format(len(data["cases"]), parameters_path))
        except Exception as exc:
            output.append_stderr("[error] reload failed: {}\n".format(exc))

    def selected_for(group):
        if state["data"] is None:
            reload_parameters()
        if group == "selected":
            return select_cases(state["data"], "all", case_name.value)
        return select_cases(state["data"], group, None)

    def worker(group):
        try:
            selected = selected_for(group)
            if real_motion.value and len(selected) > 1 and not confirm_multiple.value:
                raise RuntimeError("enable 'Allow multiple real cases' before a real group run")
            config = load_config(overrides={
                "runtime": {"dry_run": {"base": not bool(real_motion.value)}}
            })
            controller = BaseController(config)
            state["controller"] = controller
            state["stop_event"] = threading.Event()
            log("[run] started mode={} cases={}".format(
                "REAL" if real_motion.value else "DRY_RUN",
                [case["name"] for case in selected],
            ))
            result = run_cases(
                state["data"],
                selected,
                real_motion.value,
                controller=controller,
                stop_event=state["stop_event"],
                surface_label=surface_label.value,
            )
            state["last_output"] = result["path"]
            state["last_run_id"] = result["run_ids"][-1] if result["run_ids"] else None
            log("[run] finished output={} last_run_id={}".format(result["path"], state["last_run_id"]))
        except Exception as exc:
            output.append_stderr("[error] run failed: {}\n".format(exc))
        finally:
            if state["controller"] is not None:
                state["controller"].stop()
            state["controller"] = None
            state["thread"] = None

    def start(group):
        thread = state.get("thread")
        if thread is not None and thread.is_alive():
            log("[run] already running")
            return
        thread = threading.Thread(target=worker, args=(group,))
        thread.daemon = True
        state["thread"] = thread
        thread.start()

    def print_cases(_=None):
        if state["data"] is None:
            reload_parameters()
        for case in state["data"]["cases"]:
            log("{name}: motion={motion} speed={speed} seconds={seconds} repeat={repeat}".format(**case))

    def stop_base(_=None):
        state["stop_event"].set()
        controller = state.get("controller")
        if controller is not None:
            controller.stop()
        log("[stop] base stop requested")

    def record_measurement(_=None):
        try:
            if not state.get("last_output") or not state.get("last_run_id"):
                raise RuntimeError("run one case before recording a measurement")
            row = annotate_measurement(state["last_output"], state["last_run_id"], measured_value.value)
            log("[measurement] {}".format(row))
        except Exception as exc:
            output.append_stderr("[error] measurement failed: {}\n".format(exc))

    reload_button.on_click(reload_parameters)
    list_button.on_click(print_cases)
    run_case_button.on_click(lambda _: start("selected"))
    run_forward_button.on_click(lambda _: start("forward"))
    run_turn_button.on_click(lambda _: start("turn"))
    stop_button.on_click(stop_base)
    record_button.on_click(record_measurement)
    clear_button.on_click(lambda _: output.clear_output())

    panel = widgets.VBox([
        widgets.HTML("<b>Base speed x time calibration</b>"),
        widgets.HTML("Edit base_speed_time_parameters.json, then click Reload JSON."),
        widgets.HBox([reload_button, list_button, clear_button]),
        widgets.HBox([case_name, surface_label, real_motion, confirm_multiple]),
        widgets.HBox([run_case_button, run_forward_button, run_turn_button, stop_button]),
        widgets.HBox([measured_value, record_button]),
        output,
    ])
    reload_parameters()
    display(panel)
    return panel


def main():
    parser = argparse.ArgumentParser(description="Test JetBot forward/turn speed and time combinations")
    parser.add_argument("--parameters", default=DEFAULT_PARAMETERS)
    parser.add_argument("--group", choices=("forward", "turn", "all"), default="all")
    parser.add_argument("--case", help="run one named JSON case")
    parser.add_argument("--list", action="store_true", help="print configured cases without moving")
    parser.add_argument("--real", action="store_true", help="enable real base motion")
    parser.add_argument(
        "--confirm-multiple",
        action="store_true",
        help="required when running more than one real case",
    )
    args = parser.parse_args()

    data = load_parameters(os.path.abspath(args.parameters))
    selected = select_cases(data, args.group, args.case)
    if args.list:
        for case in selected:
            print("{name}: motion={motion} speed={speed} seconds={seconds} repeat={repeat}".format(**case))
        return 0
    if args.real and bool(data.get("dry_run", True)):
        print("[calibration] --real overrides JSON dry_run=true for this invocation")
    if args.real and len(selected) > 1 and not args.confirm_multiple:
        raise RuntimeError("multiple real cases require --confirm-multiple; use --case for one safe run")
    run_cases(data, selected, args.real)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[calibration] interrupted; base stop requested")
        sys.exit(130)
