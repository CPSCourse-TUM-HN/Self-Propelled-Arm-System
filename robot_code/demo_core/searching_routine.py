from __future__ import print_function

import math

from .turn_response import TurnResponseModel
from .vague_map import linear_distance_meters


class SearchingRoutine(object):
    """Validated declarative sequence of low-priority base motions."""

    def __init__(self, definition):
        self.id = int(definition["id"])
        self.name = str(definition["name"])
        self.description = str(definition.get("description", ""))
        self.start_assumption = dict(definition.get("start_assumption", {}))
        self.obstacle_aware = bool(definition.get("obstacle_aware", False))
        self.return_to_origin_after_avoidance = bool(
            definition.get("return_to_origin_after_avoidance", False)
        )
        self.repeat = bool(definition.get("repeat", False))
        self.timeout_seconds = float(definition["timeout_seconds"])
        self.steps = [dict(step) for step in definition["steps"]]


class SearchingRoutineLibrary(object):
    def __init__(self, data, config):
        validate_predefined_routines(data, config.section("base_motion_speed_profiles"))
        self.dry_run_step_seconds = float(data.get("dry_run_step_seconds", 0.2))
        self.by_id = {}
        self.by_name = {}
        for definition in data["routines"]:
            routine = SearchingRoutine(definition)
            self.by_id[routine.id] = routine
            self.by_name[routine.name] = routine

    def get(self, identifier):
        if isinstance(identifier, bool):
            raise ValueError("searching routine identifier cannot be boolean")
        if isinstance(identifier, (int, float)) and int(identifier) == identifier:
            routine = self.by_id.get(int(identifier))
        else:
            text = str(identifier)
            routine = self.by_name.get(text)
            if routine is None and text.isdigit():
                routine = self.by_id.get(int(text))
        if routine is None:
            raise ValueError("unknown searching routine: {}".format(identifier))
        return routine


def _speed_reference(value, profiles, expected_type):
    if not isinstance(value, str):
        raise ValueError("routine speed must be a symbolic {} speed reference".format(expected_type))
    parts = value.split(".")
    if len(parts) != 2 or parts[0] != expected_type or parts[1] not in ("slow", "fast"):
        raise ValueError("routine speed must reference {}.slow or {}.fast".format(expected_type, expected_type))
    return float(profiles[parts[0]][parts[1]])


def validate_predefined_routines(data, profiles):
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("predefined routines schema_version must be 1")
    dry_step = float(data.get("dry_run_step_seconds", 0.0))
    if not math.isfinite(dry_step) or dry_step <= 0.0:
        raise ValueError("predefined routines dry_run_step_seconds must be positive")
    definitions = data.get("routines")
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("predefined routines must be a non-empty list")
    ids = set()
    names = set()
    for definition in definitions:
        raw_id = definition["id"]
        if isinstance(raw_id, bool) or int(raw_id) != raw_id:
            raise ValueError("searching routine id must be an integer")
        routine_id = int(raw_id)
        name = str(definition["name"])
        if routine_id in ids or name in names:
            raise ValueError("duplicate searching routine id or name: {} / {}".format(routine_id, name))
        ids.add(routine_id)
        names.add(name)
        if "start_assumption" in definition and not isinstance(definition["start_assumption"], dict):
            raise ValueError("searching routine start_assumption must be an object")
        if bool(definition.get("return_to_origin_after_avoidance", False)) and not bool(
            definition.get("obstacle_aware", False)
        ):
            raise ValueError("return_to_origin_after_avoidance requires obstacle_aware")
        timeout = float(definition.get("timeout_seconds", 0.0))
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("searching routine timeout_seconds must be positive")
        steps = definition.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("searching routine steps must be a non-empty list")
        for index, step in enumerate(steps):
            motion = step.get("motion")
            direction = step.get("direction")
            if motion == "drive":
                if direction not in ("forward", "backward"):
                    raise ValueError("routine drive direction must be forward or backward")
                _speed_reference(step.get("speed"), profiles, "linear")
                distance = float(step.get("distance_m", 0.0))
                if not math.isfinite(distance) or distance <= 0.0:
                    raise ValueError("routine drive distance_m must be positive")
            elif motion == "turn":
                if direction not in ("left", "right"):
                    raise ValueError("routine turn direction must be left or right")
                _speed_reference(step.get("speed"), profiles, "turn")
                if "angle_rad" in step:
                    angle = float(step["angle_rad"])
                    if not math.isfinite(angle) or angle <= 0.0:
                        raise ValueError("routine turn angle_rad must be positive")
                elif len(steps) != 1 or bool(definition.get("repeat", False)):
                    raise ValueError("an unbounded turn must be the only step of a non-repeating routine")
            else:
                raise ValueError("unsupported routine motion at step {}: {}".format(index, motion))
    return True


class SearchingRoutineExecutor(object):
    """Incremental executor; target detection is intentionally owned by its caller."""

    STATE_KEY = "searching_routine"

    def __init__(self, config, base, library=None, vague_map=None):
        self.config = config
        self.base = base
        self.vague_map = vague_map
        self.library = library or SearchingRoutineLibrary(config.section("predefined_routines"), config)
        self.turn_response = TurnResponseModel(
            config.section("base_turn_response"),
            config.get("vague_map.odometry.angular_radians_per_speed_second", math.pi),
        )

    def _speed(self, reference):
        motion_type, level = str(reference).split(".")
        return float(self.config.get("base_motion_speed_profiles.{}.{}".format(motion_type, level)))

    def _effective_speed(self, direction, requested_speed):
        snapshot = self.base.command_snapshot() if hasattr(self.base, "command_snapshot") else None
        if snapshot and snapshot.get("direction") == direction:
            return float(snapshot.get("speed", requested_speed))
        scale = float(self.config.get("base_motion_command_scales.{}".format(direction), 1.0))
        return float(requested_speed) * scale

    def _progress_increment(self, step, speed, elapsed):
        if step["motion"] == "drive":
            odometry = self.config.section("vague_map")["odometry"]
            return linear_distance_meters(odometry, speed, elapsed)
        return abs(self.turn_response.angle_radians(step["direction"], speed, elapsed))

    def _target(self, step):
        return step.get("distance_m") if step["motion"] == "drive" else step.get("angle_rad")

    def _new_state(self, routine):
        origin = self.vague_map.robot_pose.as_dict() if self.vague_map is not None else None
        return {
            "identity": "{}:{}".format(routine.id, routine.name),
            "step_index": 0,
            "progress": 0.0,
            "cycles": 0,
            "phase": "route",
            "origin_pose": origin,
            "avoidance_history": [],
            "paused_for_avoidance": False,
        }

    def _state(self, routine, state_data):
        state = state_data.get(self.STATE_KEY)
        identity = "{}:{}".format(routine.id, routine.name)
        if not isinstance(state, dict) or state.get("identity") != identity:
            state = self._new_state(routine)
            state_data[self.STATE_KEY] = state
        return state

    def _capture_route_progress(self, routine, state):
        if state.get("phase") != "route":
            return 0.0
        index = int(state["step_index"])
        step = routine.steps[index]
        snapshot = self.base.command_snapshot() if hasattr(self.base, "command_snapshot") else None
        expected_suffix = "_routine_{}_step_{}".format(routine.id, index)
        label = snapshot.get("label", "") if snapshot else ""
        if not label.endswith(expected_suffix):
            return 0.0
        if not (hasattr(self.base, "motion_active") and self.base.motion_active(label)):
            return 0.0
        elapsed = (
            float(self.base.update_motion_odometry(self.library.dry_run_step_seconds))
            if hasattr(self.base, "update_motion_odometry")
            else 0.0
        )
        requested_speed = self._speed(step["speed"])
        effective_speed = (
            float(snapshot.get("speed", requested_speed))
            if snapshot and snapshot.get("direction") == step["direction"]
            else self._effective_speed(step["direction"], requested_speed)
        )
        state["progress"] = float(state.get("progress", 0.0)) + self._progress_increment(
            step, effective_speed, elapsed
        )
        return elapsed

    def current_motion_is_translation(self, routine, state_data):
        state = self._state(routine, state_data)
        if state.get("phase") == "return_position":
            if self.vague_map is None or not state.get("origin_pose"):
                return False
            pose = self.vague_map.robot_pose
            origin = state["origin_pose"]
            dx = float(origin["x_m"]) - pose.x_m
            dy = float(origin["y_m"]) - pose.y_m
            if math.hypot(dx, dy) <= float(
                self.config.get("vague_map.navigation.arrival_tolerance_m", 0.2)
            ):
                return False
            desired = math.atan2(dx, dy)
            error = ((desired - pose.heading_rad + math.pi) % (2.0 * math.pi)) - math.pi
            tolerance = float(self.config.get("vague_map.navigation.heading_tolerance_rad", 0.12))
            return abs(error) <= tolerance
        if state.get("phase") != "route":
            return False
        return routine.steps[int(state["step_index"])]["motion"] == "drive"

    def pause_for_avoidance(self, routine, state_data):
        state = self._state(routine, state_data)
        self._capture_route_progress(routine, state)
        self.base.stop()
        state["paused_for_avoidance"] = True
        state["avoidance_start_pose"] = (
            self.vague_map.robot_pose.as_dict() if self.vague_map is not None else None
        )
        return state

    def resume_after_avoidance(self, routine, state_data):
        state = self._state(routine, state_data)
        start = state.pop("avoidance_start_pose", None)
        end = self.vague_map.robot_pose.as_dict() if self.vague_map is not None else None
        if start is not None and end is not None:
            state["avoidance_history"].append({
                "dx_m": float(end["x_m"]) - float(start["x_m"]),
                "dy_m": float(end["y_m"]) - float(start["y_m"]),
                "dheading_rad": ((
                    float(end["heading_rad"]) - float(start["heading_rad"]) + math.pi
                ) % (2.0 * math.pi)) - math.pi,
            })
        state["paused_for_avoidance"] = False
        return state

    def _finish_cycle(self, routine, state):
        state["cycles"] = int(state.get("cycles", 0)) + 1
        if routine.return_to_origin_after_avoidance and state.get("avoidance_history") and self.vague_map is not None:
            state["phase"] = "return_position"
            return False
        if not routine.repeat:
            return True
        replacement = self._new_state(routine)
        replacement["cycles"] = state["cycles"]
        state.clear()
        state.update(replacement)
        return False

    def _finish_recovery(self, routine, state):
        if not routine.repeat:
            return True
        cycles = int(state.get("cycles", 0))
        replacement = self._new_state(routine)
        replacement["cycles"] = cycles
        state.clear()
        state.update(replacement)
        return False

    def _recovery_step(self, routine, state, label_prefix):
        if self.vague_map is None or not state.get("origin_pose"):
            return self._finish_recovery(routine, state)
        label = "{}_routine_{}_return".format(label_prefix, routine.id)
        if hasattr(self.base, "motion_active") and self.base.motion_active():
            if hasattr(self.base, "update_motion_odometry"):
                self.base.update_motion_odometry(self.library.dry_run_step_seconds)
        pose = self.vague_map.robot_pose
        origin = state["origin_pose"]
        dx = float(origin["x_m"]) - pose.x_m
        dy = float(origin["y_m"]) - pose.y_m
        navigation = self.config.section("vague_map")["navigation"]
        if state["phase"] == "return_position":
            distance = math.hypot(dx, dy)
            if distance <= float(navigation["arrival_tolerance_m"]):
                self.base.stop()
                state["phase"] = "return_heading"
                return False
            desired = math.atan2(dx, dy)
            error = ((desired - pose.heading_rad + math.pi) % (2.0 * math.pi)) - math.pi
            if abs(error) > float(navigation["heading_tolerance_rad"]):
                self.base.start_motion(
                    "right" if error > 0.0 else "left",
                    float(navigation["turn_speed"]),
                    label + "_turn",
                )
            else:
                self.base.start_motion("forward", float(navigation["forward_speed"]), label + "_forward")
            return False
        heading_error = ((
            float(origin["heading_rad"]) - pose.heading_rad + math.pi
        ) % (2.0 * math.pi)) - math.pi
        if abs(heading_error) <= float(navigation["heading_tolerance_rad"]):
            self.base.stop()
            return self._finish_recovery(routine, state)
        self.base.start_motion(
            "right" if heading_error > 0.0 else "left",
            float(navigation["turn_speed"]),
            label + "_heading",
        )
        return False

    def step(self, routine, state_data, label_prefix):
        state = self._state(routine, state_data)
        if state.get("phase") in ("return_position", "return_heading"):
            return self._recovery_step(routine, state, label_prefix)

        index = int(state["step_index"])
        step = routine.steps[index]
        label = "{}_routine_{}_step_{}".format(label_prefix, routine.id, index)
        self._capture_route_progress(routine, state)

        target = self._target(step)
        if target is not None and float(state["progress"]) >= float(target):
            self.base.stop()
            state["step_index"] = index + 1
            state["progress"] = 0.0
            if state["step_index"] >= len(routine.steps):
                return self._finish_cycle(routine, state)
            return False

        requested_speed = self._speed(step["speed"])
        self.base.start_motion(step["direction"], requested_speed, label)
        return False
