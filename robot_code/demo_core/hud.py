from __future__ import print_function

import csv
import math
import time

import cv2


EVENT_FIELDS = (
    "timestamp",
    "video_elapsed_seconds",
    "state",
    "subphase",
    "target",
    "state_elapsed_seconds",
    "state_timeout_seconds",
    "transition_event",
    "transition_reason",
    "base_active",
    "base_label",
    "base_direction",
    "base_requested_speed",
    "base_effective_speed",
    "target_found",
    "confidence",
    "error_x",
    "bbox_height_norm",
    "target_distance",
    "raw_yaw_deg",
    "yaw_control_error_deg",
    "yaw_target_deg",
    "depth_mean",
    "depth_min",
    "depth_max",
    "completed_pickups",
    "grabbed",
    "retry_count",
    "camera_available",
    "depth_available",
    "paused",
    "stop_requested",
    "last_error",
)


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _active_base_command(base):
    if hasattr(base, "command_snapshot"):
        motion = base.command_snapshot()
        lock = None
    else:
        lock = getattr(base, "_motion_lock", None)
    if not hasattr(base, "command_snapshot") and lock is None:
        motion = getattr(base, "_active_motion", None)
    elif not hasattr(base, "command_snapshot"):
        with lock:
            motion = getattr(base, "_active_motion", None)
            motion = dict(motion) if motion else None
    if not motion:
        return {
            "base_active": False,
            "base_label": "stop",
            "base_direction": "stop",
            "base_requested_speed": 0.0,
            "base_effective_speed": 0.0,
        }
    return {
        "base_active": True,
        "base_label": motion.get("label"),
        "base_direction": motion.get("direction"),
        "base_requested_speed": _finite(motion.get("requested_speed", motion.get("speed"))),
        "base_effective_speed": _finite(motion.get("speed")),
    }


def _state_timeout(runtime):
    state = runtime.state.value
    target = runtime.context.target_type
    try:
        if state == "SEARCHING" and target is not None:
            search = runtime.config.target_navigation(target)["search"]
            routine_id = search.get("routine")
            if (
                getattr(target, "value", None) == "bin"
                and getattr(runtime.previous_state, "value", None) == "MAP_NAVIGATING"
            ):
                routine_id = search.get("map_arrival_routine", routine_id)
            return _finite(runtime.navigator.searching_routines.library.get(routine_id).timeout_seconds)
        if state == "APPROACHING" and target is not None:
            return _finite(runtime.config.target_navigation(target)["approach"].get("timeout_seconds"))
        if state == "MAP_NAVIGATING":
            return _finite(runtime.config.get("vague_map.navigation.timeout_seconds"))
        if state == "BIN_SIDE_DOCKING":
            return _finite(runtime.config.get("navigation.bin.side_docking.experimental.timeout_seconds"))
        if state == "AVOIDING":
            return _finite(runtime.config.get("avoidance.timeout_seconds"))
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _subphase(runtime, base_command):
    action = getattr(runtime, "current_action", None)
    if action:
        return str(action)
    if base_command["base_active"]:
        return str(base_command["base_label"])
    state = runtime.state.value
    data = runtime.context.state_data
    if state == "BIN_SIDE_DOCKING":
        return "geometry_alignment" if data.get("entry_complete") else "side_view_entry"
    if state == "FINAL_VERIFY":
        return "stable_{}/?".format(int(data.get("stable_frames", 0)))
    if state == "FINALIZING":
        return "pickup_sequence" if getattr(runtime.context.target_type, "value", None) == "can" else "release_sequence"
    return "idle" if state == "IDLE" else "observe"


def runtime_hud_snapshot(runtime, depth_stats=None, frame_available=True, recording=False, now=None):
    """Return a JSON/CSV-friendly, read-only view of the current mission."""
    now = float(now if now is not None else time.time())
    context = runtime.context
    observation = dict(context.last_observation or {})
    transition = dict(getattr(runtime, "last_transition", None) or {})
    base = _active_base_command(runtime.services.base)
    pose = observation.get("pose") or {}
    raw_yaw_rad = _finite(observation.get("raw_yaw_error_rad"))
    raw_yaw_deg = (
        math.degrees(raw_yaw_rad)
        if raw_yaw_rad is not None
        else _finite(observation.get("yaw_error_deg"))
    )
    control_yaw_rad = _finite(observation.get("yaw_error_calibrated_rad"))
    target_yaw_rad = _finite(observation.get("yaw_target_rad"))
    snapshot = {
        "timestamp": now,
        "state": runtime.state.value,
        "target": context.target_type.value if context.target_type is not None else "none",
        "state_elapsed_seconds": _finite(context.elapsed()),
        "state_timeout_seconds": _state_timeout(runtime),
        "transition_event": transition.get("event"),
        "transition_reason": transition.get("reason"),
        "transition_timestamp": _finite(transition.get("timestamp")),
        "target_found": bool(observation.get("found", context.target_found)),
        "confidence": _finite(observation.get("confidence")),
        "error_x": _finite(observation.get("error_x")),
        "bbox_height_norm": _finite(observation.get("bbox_height_norm")),
        "bbox": observation.get("bbox"),
        "center_x": _finite(observation.get("center_x")),
        "center_y": _finite(observation.get("center_y")),
        "target_distance": _finite(observation.get("distance", context.distance_to_target)),
        "raw_yaw_deg": raw_yaw_deg,
        "yaw_control_error_deg": (
            None if control_yaw_rad is None else math.degrees(control_yaw_rad)
        ),
        "yaw_target_deg": None if target_yaw_rad is None else math.degrees(target_yaw_rad),
        "pose_x": _finite(pose.get("x")),
        "pose_y": _finite(pose.get("y")),
        "pose_z": _finite(pose.get("z")),
        "depth_mean": _finite((depth_stats or {}).get("mean")),
        "depth_min": _finite((depth_stats or {}).get("min")),
        "depth_max": _finite((depth_stats or {}).get("max")),
        "completed_pickups": int(context.completed_pickups),
        "grabbed": bool(context.grabbed),
        "retry_count": sum(int(value) for value in context.retry_counts.values()),
        "camera_available": bool(frame_available),
        "depth_available": bool(depth_stats),
        "paused": bool(runtime.pause_requested),
        "stop_requested": bool(runtime.stop_requested),
        "recording": bool(recording),
        "last_error": context.last_error,
    }
    snapshot.update(base)
    snapshot["subphase"] = _subphase(runtime, base)
    return snapshot


def _text(canvas, text, y, color=(255, 255, 255), scale=0.52):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    size, baseline = cv2.getTextSize(str(text), font, scale, thickness)
    cv2.rectangle(canvas, (6, y - size[1] - 6), (14 + size[0], y + baseline + 3), (20, 20, 20), -1)
    cv2.putText(canvas, str(text), (10, y), font, scale, color, thickness, cv2.LINE_AA)


def _fmt(value, digits=3):
    return "n/a" if value is None else ("{:.%df}" % digits).format(float(value))


def render_hud(frame, snapshot, camera_settings=None, enabled=True, size=(640, 480)):
    """Resize a BGR frame and render the compact demo HUD without mutating it."""
    source_height, source_width = frame.shape[:2]
    width, height = int(size[0]), int(size[1])
    canvas = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    if not enabled:
        return canvas
    scale_x = width / float(max(1, source_width))
    scale_y = height / float(max(1, source_height))
    center_x, center_y = width // 2, height // 2
    cv2.line(canvas, (center_x, 0), (center_x, height - 1), (150, 150, 150), 1, cv2.LINE_AA)
    camera_settings = camera_settings or {}
    roi_w = float(camera_settings.get("lens_roi_width", 0.12))
    roi_h = float(camera_settings.get("lens_roi_height", 0.18))
    rx, ry = int(width * roi_w / 2.0), int(height * roi_h / 2.0)
    cv2.rectangle(canvas, (center_x - rx, center_y - ry), (center_x + rx, center_y + ry), (255, 180, 0), 2)
    box = snapshot.get("bbox")
    if box and len(box) == 4:
        x1, y1, x2, y2 = box
        cv2.rectangle(canvas, (int(float(x1) * scale_x), int(float(y1) * scale_y)), (int(float(x2) * scale_x), int(float(y2) * scale_y)), (0, 255, 0), 2)
    if snapshot.get("center_x") is not None and snapshot.get("center_y") is not None:
        point = (int(snapshot["center_x"] * scale_x), int(snapshot["center_y"] * scale_y))
        cv2.circle(canvas, point, 6, (0, 255, 0), -1, cv2.LINE_AA)

    elapsed = _fmt(snapshot.get("state_elapsed_seconds"), 1)
    timeout = _fmt(snapshot.get("state_timeout_seconds"), 1)
    _text(canvas, "{} | {} | {} | {}/{}s".format(snapshot["state"], snapshot["subphase"], snapshot["target"], elapsed, timeout), 23)
    _text(canvas, "BASE {} req={} eff={} {}".format(snapshot["base_direction"], _fmt(snapshot.get("base_requested_speed"), 2), _fmt(snapshot.get("base_effective_speed"), 2), snapshot["base_label"]), 46, (100, 230, 255))
    _text(canvas, "TARGET found={} conf={} err_x={} h={} dist={}".format(snapshot["target_found"], _fmt(snapshot.get("confidence")), _fmt(snapshot.get("error_x")), _fmt(snapshot.get("bbox_height_norm")), _fmt(snapshot.get("target_distance"))), 69, (0, 255, 0) if snapshot["target_found"] else (150, 150, 150))
    if snapshot.get("state") == "BIN_SIDE_DOCKING":
        _text(
            canvas,
            "TAG YAW raw={} ctrl={} target={} deg".format(
                _fmt(snapshot.get("raw_yaw_deg"), 1),
                _fmt(snapshot.get("yaw_control_error_deg"), 1),
                _fmt(snapshot.get("yaw_target_deg"), 1),
            ),
            92,
            (220, 180, 255),
        )
    else:
        _text(canvas, "DEPTH mean={} min={} max={}".format(_fmt(snapshot.get("depth_mean")), _fmt(snapshot.get("depth_min")), _fmt(snapshot.get("depth_max"))), 92, (255, 210, 0) if snapshot["depth_available"] else (150, 150, 150))
    _text(canvas, "MISSION pickups={} grabbed={} retry={} CAM={} DEPTH={}".format(snapshot["completed_pickups"], snapshot["grabbed"], snapshot["retry_count"], "OK" if snapshot["camera_available"] else "NO", "OK" if snapshot["depth_available"] else "NO"), 115)
    if snapshot.get("pose_z") is not None:
        _text(canvas, "TAG PnP x={} y={} z={}".format(_fmt(snapshot.get("pose_x")), _fmt(snapshot.get("pose_y")), _fmt(snapshot.get("pose_z"))), 138, (220, 180, 255))
    if snapshot.get("recording"):
        cv2.circle(canvas, (width - 25, 22), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, "REC", (width - 82, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    banner = None
    banner_color = (0, 170, 255)
    transition_age = None
    if snapshot.get("transition_timestamp") is not None:
        transition_age = snapshot["timestamp"] - snapshot["transition_timestamp"]
    if snapshot["state"] == "FAILED":
        banner = "ERROR: {}".format(snapshot.get("last_error") or snapshot.get("transition_reason") or "mission failed")
        banner_color = (0, 0, 220)
    elif transition_age is not None and transition_age <= 4.0 and snapshot.get("transition_event"):
        banner = "EVENT {}{}".format(snapshot["transition_event"], ": " + str(snapshot["transition_reason"]) if snapshot.get("transition_reason") else "")
    if banner:
        cv2.rectangle(canvas, (0, height - 36), (width - 1, height - 1), banner_color, -1)
        cv2.putText(canvas, banner[:92], (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


class HudEventRecorder(object):
    """Write one synchronized telemetry row for each recorded video frame."""

    def __init__(self, path):
        self.path = str(path)
        self._stream = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._stream, fieldnames=EVENT_FIELDS, extrasaction="ignore")
        self._writer.writeheader()
        self._stream.flush()

    def write(self, snapshot, video_elapsed_seconds):
        row = dict(snapshot)
        row["video_elapsed_seconds"] = _finite(video_elapsed_seconds)
        self._writer.writerow(row)
        self._stream.flush()

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None
