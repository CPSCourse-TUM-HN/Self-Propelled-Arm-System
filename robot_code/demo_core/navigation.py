from __future__ import print_function

import math
import time

from .fsm_types import MissionEvent, TargetType
from .searching_routine import SearchingRoutineExecutor


APPROACH_STOP_MODES = ("bbox_height", "depth", "bbox_height_and_depth")


def evaluate_approach_stop(stop, bbox_height_norm=None, depth_raw=None, pnp_z_m=None):
    """Evaluate one visual/depth/PnP stop sample without performing hardware I/O."""
    mode = str(stop.get("mode", "bbox_height"))
    if mode not in APPROACH_STOP_MODES + ("pnp_z",):
        raise ValueError("unsupported approach stop mode: {}".format(mode))
    legacy_threshold = stop.get("threshold")
    bbox_threshold = stop.get("bbox_height_threshold", legacy_threshold)
    depth_threshold = stop.get("depth_raw_threshold", legacy_threshold)
    pnp_z_threshold = stop.get("pnp_z_threshold_m", legacy_threshold)
    bbox_value = None if bbox_height_norm is None else float(bbox_height_norm)
    depth_value = None if depth_raw is None else float(depth_raw)
    pnp_z_value = None if pnp_z_m is None else float(pnp_z_m)
    bbox_reached = (
        bbox_value is not None
        and bbox_threshold is not None
        and bbox_value >= float(bbox_threshold)
    )
    depth_reached = (
        depth_value is not None
        and depth_threshold is not None
        and depth_value <= float(depth_threshold)
    )
    pnp_z_reached = (
        pnp_z_value is not None
        and math.isfinite(pnp_z_value)
        and pnp_z_value > 0.0
        and pnp_z_threshold is not None
        and pnp_z_value <= float(pnp_z_threshold)
    )
    if mode == "bbox_height":
        reached = bbox_reached
    elif mode == "depth":
        reached = depth_reached
    elif mode == "bbox_height_and_depth":
        reached = bbox_reached and depth_reached
    else:
        reached = pnp_z_reached
    return {
        "mode": mode,
        "reached": bool(reached),
        "bbox_height_norm": bbox_value,
        "bbox_height_threshold": None if bbox_threshold is None else float(bbox_threshold),
        "bbox_reached": bool(bbox_reached),
        "depth_raw": depth_value,
        "depth_raw_threshold": None if depth_threshold is None else float(depth_threshold),
        "depth_reached": bool(depth_reached),
        "pnp_z_m": pnp_z_value,
        "pnp_z_threshold_m": None if pnp_z_threshold is None else float(pnp_z_threshold),
        "pnp_z_reached": bool(pnp_z_reached),
    }


class StepOutcome(object):
    def __init__(self, event=None, observation=None, reason=None):
        self.event = event
        self.observation = observation
        self.reason = reason


class TargetNavigator(object):
    """Incremental navigation shared by can and bin targets."""

    def __init__(self, config, context, base, depth, can_detector, bin_detector, frame_observer=None, vague_map=None):
        self.config = config
        self.context = context
        self.base = base
        self.depth = depth
        self.can_detector = can_detector
        self.bin_detector = bin_detector
        self.frame_observer = frame_observer
        self.searching_routines = (
            SearchingRoutineExecutor(config, base, vague_map=vague_map)
            if hasattr(config, "section")
            else None
        )

    def detector(self, target_type):
        return self.can_detector if target_type == TargetType.CAN else self.bin_detector

    def detect(self, target_type, frame=None):
        if frame is None:
            frame = self.depth.read_frame()
        if frame is None:
            return {"found": False, "reason": "no_frame"}
        if self.frame_observer is not None:
            self.frame_observer(target_type, frame)
        start = time.time()
        observation = self.detector(target_type).detect(frame)
        self.context.record_detection(target_type, time.time() - start)
        return observation

    def _accepted(self, target_type, observation, tracking=False):
        if not observation or not observation.get("found"):
            return False
        if target_type != TargetType.CAN:
            return True
        threshold = self.can_detector.confidence_threshold(tracking)
        return float(observation.get("confidence", 0.0)) >= threshold

    def _start_visual_turn(self, direction, configured_speed, label):
        speed = float(self.config.get("base_motion_speed_profiles.turn.slow", configured_speed))
        if self.base.start_motion(direction, speed, label):
            print("[visual-turn] label={} direction={} speed={}".format(label, direction, speed))

    def search_step(self, target_type, routine_identifier=None):
        if self.searching_routines is None:
            raise RuntimeError("searching routines require a full DemoConfig")
        settings = self.config.target_navigation(target_type)["search"]
        frame = self.depth.read_frame()
        observation = self.detect(target_type, frame)
        if self._accepted(target_type, observation, tracking=False):
            self.base.stop()
            return StepOutcome(MissionEvent.TARGET_FOUND, observation)
        routine = self.searching_routines.library.get(
            settings["routine"] if routine_identifier is None else routine_identifier
        )
        if self.context.elapsed() >= routine.timeout_seconds:
            self.base.stop()
            return StepOutcome(MissionEvent.TIMEOUT, reason="search routine timeout")
        routine_data = self.context.searching_routine_data
        if (
            routine.obstacle_aware
            and self.config.get("avoidance.strategy", "disabled") != "disabled"
            and self.searching_routines.current_motion_is_translation(routine, routine_data)
            and frame is not None
            and self.depth.obstacle_detected_frame(frame)
        ):
            self.searching_routines.pause_for_avoidance(routine, routine_data)
            self.context.obstacle_found = True
            return StepOutcome(MissionEvent.OBSTACLE_FOUND, observation, "search routine obstacle")
        complete = self.searching_routines.step(
            routine,
            routine_data,
            "search_{}".format(target_type.value),
        )
        self.context.increment_step()
        if complete:
            self.base.stop()
            return StepOutcome(MissionEvent.TIMEOUT, observation, "search routine complete")
        return StepOutcome(observation=observation)

    def align_step(self, target_type, settings=None):
        settings = settings or self.config.target_navigation(target_type)["align"]
        if self.context.increment_step() > int(settings["max_steps"]):
            self.base.stop()
            return StepOutcome(MissionEvent.TIMEOUT, reason="alignment max steps")
        observation = self.detect(target_type)
        if not self._accepted(target_type, observation, tracking=True):
            lost = int(self.context.state_data.get("lost_frames", 0)) + 1
            self.context.state_data["lost_frames"] = lost
            if lost >= int(settings.get("lost_frame_limit", 5)):
                self.base.stop()
                return StepOutcome(MissionEvent.TARGET_MISSING, observation, "alignment target lost")
            label = "align_{}".format(target_type.value)
            grace_frames = int(settings.get("lost_motion_grace_frames", 0))
            keep_turning = lost <= grace_frames and self.base.motion_active(label)
            if keep_turning:
                print(
                    "[visual-turn] label={} target temporarily missing; holding motion frame={}/{}".format(
                        label, lost, grace_frames,
                    )
                )
            else:
                self.base.stop()
            time.sleep(float(self.config.get("camera.observation_pause_seconds", 0.1)))
            return StepOutcome(observation=observation)
        self.context.state_data["lost_frames"] = 0
        error_x = float(observation["error_x"])
        target_error_x = float(settings.get("target_error_x_norm", 0.0))
        alignment_error_x = error_x - target_error_x
        observation["alignment_target_error_x"] = target_error_x
        observation["alignment_error_x"] = alignment_error_x
        if abs(alignment_error_x) <= float(settings["tolerance_norm"]):
            self.base.stop()
            return StepOutcome(MissionEvent.TARGET_ALIGNED, observation)
        direction = "right" if alignment_error_x > 0 else "left"
        self._start_visual_turn(direction, settings["speed"], "align_{}".format(target_type.value))
        return StepOutcome(observation=observation)

    def _depth_stop_value(self, frame):
        stats = self.depth.observe_lens_center_frame(frame)
        if not stats:
            return None
        raw = float(stats["mean"])
        self.context.distance_to_target = raw
        return raw

    def _stop_status(self, frame, observation, stop):
        mode = str(stop.get("mode", "bbox_height"))
        depth_raw = self._depth_stop_value(frame) if mode in ("depth", "bbox_height_and_depth") else None
        pose = (observation or {}).get("pose") or {}
        pnp_z_m = pose.get("z") if mode == "pnp_z" else None
        try:
            pnp_z_m = float(pnp_z_m)
            if not math.isfinite(pnp_z_m) or pnp_z_m <= 0.0:
                pnp_z_m = None
        except (TypeError, ValueError):
            pnp_z_m = None
        return evaluate_approach_stop(
            stop,
            bbox_height_norm=observation.get("bbox_height_norm") if observation else None,
            depth_raw=depth_raw,
            pnp_z_m=pnp_z_m,
        )

    def approach_step(self, target_type):
        settings = self.config.target_navigation(target_type)["approach"]
        forward_label = "approach_{}_forward".format(target_type.value)
        if self.context.elapsed() >= float(settings["timeout_seconds"]):
            self.base.stop()
            return StepOutcome(MissionEvent.TIMEOUT, reason="approach timeout")
        if self.context.increment_step() > int(settings.get("max_pulses", 0) or 1000000):
            self.base.stop()
            return StepOutcome(MissionEvent.TIMEOUT, reason="approach max pulses")
        frame = self.depth.read_frame()
        if frame is None:
            self.base.stop()
            return StepOutcome(MissionEvent.FAIL, reason="camera frame unavailable")
        if self.config.get("avoidance.strategy", "disabled") != "disabled" and self.depth.obstacle_detected_frame(frame):
            self.base.stop()
            self.context.obstacle_found = True
            return StepOutcome(MissionEvent.OBSTACLE_FOUND, reason="depth obstacle")
        observation = self.detect(target_type, frame)
        if not self._accepted(target_type, observation, tracking=True):
            lost = int(self.context.state_data.get("lost_frames", 0)) + 1
            self.context.state_data["lost_frames"] = lost
            self.base.stop()
            if lost >= int(settings.get("lost_frame_limit", 4)):
                return StepOutcome(MissionEvent.TARGET_MISSING, observation, "approach target lost")
            time.sleep(float(self.config.get("camera.observation_pause_seconds", 0.1)))
            return StepOutcome(observation=observation)
        stop = settings["stop"]
        stop_status = self._stop_status(frame, observation, stop)
        if stop_status["mode"] == "pnp_z" and stop_status["pnp_z_m"] is None:
            lost = int(self.context.state_data.get("lost_frames", 0)) + 1
            self.context.state_data["lost_frames"] = lost
            self.base.stop()
            if lost >= int(settings.get("lost_frame_limit", 4)):
                return StepOutcome(MissionEvent.TARGET_MISSING, observation, "bin PnP pose unavailable")
            time.sleep(float(self.config.get("camera.observation_pause_seconds", 0.1)))
            return StepOutcome(observation=observation, reason="waiting for calibrated bin PnP pose")
        self.context.state_data["lost_frames"] = 0
        if stop_status["pnp_z_m"] is not None:
            self.context.distance_to_target = stop_status["pnp_z_m"]
            observation["pnp_forward_distance_m"] = stop_status["pnp_z_m"]
            observation["pnp_stop_threshold_m"] = stop_status["pnp_z_threshold_m"]
        print(
            "[approach] target={} mode={} bbox={}/{} bbox_ok={} depth_raw={}/{} depth_ok={} "
            "pnp_z={}/{} pnp_ok={} error_x={:.3f}".format(
                target_type.value,
                stop_status["mode"],
                "{:.3f}".format(stop_status["bbox_height_norm"]) if stop_status["bbox_height_norm"] is not None else "n/a",
                stop_status["bbox_height_threshold"],
                stop_status["bbox_reached"],
                "{:.3f}".format(stop_status["depth_raw"]) if stop_status["depth_raw"] is not None else "n/a",
                stop_status["depth_raw_threshold"],
                stop_status["depth_reached"],
                "{:.3f}".format(stop_status["pnp_z_m"]) if stop_status["pnp_z_m"] is not None else "n/a",
                stop_status["pnp_z_threshold_m"],
                stop_status["pnp_z_reached"],
                float(observation["error_x"]),
            )
        )
        if stop_status["reached"]:
            self.base.stop()
            return StepOutcome(MissionEvent.TARGET_REACHED, observation)
        error_x = float(observation["error_x"])
        if abs(error_x) > float(settings.get("steering_tolerance_norm", 1.0)):
            direction = "right" if error_x > 0 else "left"
            self._start_visual_turn(
                direction,
                settings.get("steering_speed", settings["speed"]),
                "approach_{}_steer".format(target_type.value),
            )
        else:
            self.base.start_motion("forward", float(settings["speed"]), forward_label)
        return StepOutcome(observation=observation)

    def final_verify_step(self, target_type):
        navigation = self.config.target_navigation(target_type)
        settings = dict(navigation["align"])
        settings.update(navigation.get("near_align", {}))
        if not bool(settings.get("enabled", True)):
            return StepOutcome(MissionEvent.TARGET_STABLE, self.context.last_observation)
        outcome = self.align_step(target_type, settings=settings)
        if outcome.event == MissionEvent.TARGET_ALIGNED:
            stable = int(self.context.state_data.get("stable_frames", 0)) + 1
            self.context.state_data["stable_frames"] = stable
            required = int(settings["required_stable_frames"])
            if stable >= required:
                return StepOutcome(MissionEvent.TARGET_STABLE, outcome.observation)
            return StepOutcome(observation=outcome.observation)
        self.context.state_data["stable_frames"] = 0
        return outcome


class BinSideDockingNavigator(object):
    """Incremental right-facing AprilTag calibration used by exp2."""

    def __init__(self, config, context, base, depth, detector):
        self.config = config
        self.context = context
        self.base = base
        self.depth = depth
        self.detector = detector
    def _correct_yaw(self, yaw_error, settings):
        direction = (
            settings["yaw_positive_turn_direction"]
            if yaw_error > 0.0
            else self._opposite(settings["yaw_positive_turn_direction"])
        )
        if settings.get("correction_mode", "continuous") == "pulse":
            seconds = float(settings["yaw_correction_seconds"])
            self.base.stop()
            self.base.pulse(
                direction,
                float(settings["yaw_turn_speed"]),
                seconds,
                "exp2_yaw_align",
            )
            settle_seconds = float(settings.get("post_correction_camera_settle_seconds", 0.0))
            self.context.state_data["exp2_camera_ready_at"] = time.monotonic() + settle_seconds
            print(
                "[exp2] pulse yaw correction direction={} seconds={:.3f}".format(
                    direction, seconds
                )
            )
            return
        self.base.start_motion(
            direction,
            float(settings["yaw_turn_speed"]),
            "exp2_yaw_align",
        )
        print("[exp2] continuous yaw correction direction={}".format(direction))

    @staticmethod
    def _opposite(direction):
        return {
            "left": "right",
            "right": "left",
            "forward": "backward",
            "backward": "forward",
        }[direction]

    def step(self, settings):
        if self.context.elapsed() >= float(settings["timeout_seconds"]):
            self.base.stop()
            return StepOutcome(MissionEvent.TIMEOUT, reason="exp2 side docking timeout")
        if self.context.increment_step() > int(settings["max_steps"]):
            self.base.stop()
            return StepOutcome(MissionEvent.TIMEOUT, reason="exp2 side docking max steps")

        camera_ready_at = float(self.context.state_data.get("exp2_camera_ready_at", 0.0))
        if time.monotonic() < camera_ready_at:
            self.base.stop()
            return StepOutcome(reason="exp2 waiting for camera to settle")

        frame = self.depth.read_frame()
        if frame is None:
            self.base.stop()
            return StepOutcome(MissionEvent.FAIL, reason="exp2 camera frame unavailable")
        observation = self.detector.detect(frame)
        if not observation or not observation.get("found"):
            self.context.state_data["stable_frames"] = 0
            self.context.state_data["exp2_next_axis"] = "yaw"
            self.context.state_data["exp2_axis_motion_active"] = False
            self.context.state_data.pop("exp2_yaw_correction_started_at", None)
            self.context.state_data.pop("exp2_center_guard_active", None)
            lost = int(self.context.state_data.get("lost_frames", 0)) + 1
            self.context.state_data["lost_frames"] = lost
            self.base.stop()
            if lost >= int(settings.get("lost_frame_limit", 8)):
                return StepOutcome(MissionEvent.TARGET_MISSING, observation, "exp2 side tag lost")
            time.sleep(float(self.config.get("camera.observation_pause_seconds", 0.1)))
            return StepOutcome(observation=observation)

        self.context.state_data["lost_frames"] = 0
        center_error = float(observation.get("error_x", 0.0))
        raw_yaw_error = observation.get("yaw_error_rad")
        raw_yaw_error = None if raw_yaw_error is None else float(raw_yaw_error)
        yaw_reference = float(settings.get("target_yaw_rad", 0.0))
        yaw_error = None
        if raw_yaw_error is not None:
            yaw_error = math.atan2(
                math.sin(raw_yaw_error - yaw_reference),
                math.cos(raw_yaw_error - yaw_reference),
            )
            observation["raw_yaw_error_rad"] = raw_yaw_error
            observation["yaw_target_rad"] = yaw_reference
            observation["yaw_target_deg"] = math.degrees(yaw_reference)
            observation["yaw_error_calibrated_rad"] = yaw_error
            observation["yaw_error_calibrated_deg"] = math.degrees(yaw_error)
        next_axis = self.context.state_data.get("exp2_next_axis", "yaw")
        correction_mode = settings.get("correction_mode", "continuous")
        phase = next_axis
        if correction_mode == "pulse":
            phase = self.context.state_data.get("exp2_pulse_stage", "coarse_yaw")
        print(
            "[exp2] phase={} center_error={:.3f} raw_yaw_deg={} calibrated_yaw_deg={} target_yaw_deg={:.2f} "
            "yaw_source={}".format(
                "{}_{}".format(correction_mode, phase),
                center_error,
                None if raw_yaw_error is None else round(math.degrees(raw_yaw_error), 2),
                None if yaw_error is None else round(math.degrees(yaw_error), 2),
                math.degrees(yaw_reference),
                observation.get("yaw_source"),
            )
        )

        if yaw_error is None:
            self.context.state_data["stable_frames"] = 0
            self.context.state_data["exp2_next_axis"] = "yaw"
            self.context.state_data["exp2_pulse_stage"] = "coarse_yaw"
            self.context.state_data.pop("exp2_yaw_correction_started_at", None)
            self.context.state_data.pop("exp2_center_guard_active", None)
            self.base.stop()
            return StepOutcome(observation=observation, reason="exp2 Tag yaw unavailable")

        center_out = abs(center_error) > float(settings["center_tolerance_norm"])
        yaw_magnitude = abs(yaw_error)
        yaw_out = yaw_magnitude > float(settings["yaw_tolerance_rad"])

        if correction_mode == "pulse" and (center_out or yaw_out):
            self.context.state_data["stable_frames"] = 0
            self.context.state_data["exp2_axis_motion_active"] = False
            stage = self.context.state_data.get("exp2_pulse_stage", "coarse_yaw")
            self.context.state_data["exp2_pulse_stage"] = stage
            entry_yaw = float(settings["center_stage_entry_yaw_rad"])
            center_guard = float(settings["coarse_yaw_center_guard_norm"])

            if stage == "coarse_yaw" and yaw_magnitude <= entry_yaw:
                stage = "center"
                self.context.state_data["exp2_pulse_stage"] = stage
                print("[exp2] pulse stage coarse_yaw->center")

            if stage == "coarse_yaw":
                if abs(center_error) > center_guard:
                    correction_axis = "center"
                    print(
                        "[exp2] coarse yaw center guard error={:.3f} limit={:.3f}".format(
                            center_error, center_guard,
                        )
                    )
                else:
                    correction_axis = "yaw"
            elif stage == "center":
                if center_out:
                    correction_axis = "center"
                else:
                    stage = "final_yaw"
                    self.context.state_data["exp2_pulse_stage"] = stage
                    correction_axis = "yaw"
                    print("[exp2] pulse stage center->final_yaw")
            else:
                if yaw_out:
                    correction_axis = "yaw"
                else:
                    stage = "center"
                    self.context.state_data["exp2_pulse_stage"] = stage
                    correction_axis = "center"
                    print("[exp2] pulse stage final_yaw->center")

            self.context.state_data["exp2_next_axis"] = correction_axis

            if correction_axis == "yaw":
                self._correct_yaw(yaw_error, settings)
                print("[exp2] pulse stage={} axis=yaw".format(stage))
                return StepOutcome(observation=observation)

            direction = (
                settings["center_positive_drive_direction"]
                if center_error > 0.0
                else self._opposite(settings["center_positive_drive_direction"])
            )
            correction_seconds = float(settings["center_correction_seconds"])
            self.base.stop()
            self.base.pulse(
                direction,
                float(settings["drive_speed"]),
                correction_seconds,
                "exp2_center_align",
            )
            settle_seconds = float(settings.get("post_correction_camera_settle_seconds", 0.0))
            self.context.state_data["exp2_camera_ready_at"] = time.monotonic() + settle_seconds
            print(
                "[exp2] pulse center correction direction={} seconds={:.3f}; stage={}".format(
                    direction, correction_seconds, stage
                )
            )
            return StepOutcome(observation=observation)

        if correction_mode != "pulse" and next_axis == "yaw":
            center_guard = float(settings["coarse_yaw_center_guard_norm"])
            if abs(center_error) > center_guard:
                self.context.state_data["stable_frames"] = 0
                self.context.state_data["exp2_axis_motion_active"] = False
                self.context.state_data["exp2_center_guard_active"] = True
                self.context.state_data.pop("exp2_yaw_correction_started_at", None)
                self.context.state_data["exp2_next_axis"] = "center"
                self.base.stop()
                settle_seconds = float(settings.get("post_correction_camera_settle_seconds", 0.0))
                self.context.state_data["exp2_camera_ready_at"] = time.monotonic() + settle_seconds
                print(
                    "[exp2] continuous x guard preempted yaw error={:.3f} limit={:.3f}; "
                    "camera_settle_seconds={:.3f}".format(
                        center_error, center_guard, settle_seconds,
                    )
                )
                return StepOutcome(observation=observation, reason="exp2 x guard preempted yaw")

        if next_axis == "yaw":
            if yaw_out:
                self.context.state_data["stable_frames"] = 0
                now = time.monotonic()
                started_at = self.context.state_data.get("exp2_yaw_correction_started_at")
                if started_at is None:
                    started_at = now
                    self.context.state_data["exp2_yaw_correction_started_at"] = started_at
                maximum_seconds = float(settings["yaw_single_correction_max_seconds"])
                if now - float(started_at) >= maximum_seconds:
                    self.context.state_data["exp2_axis_motion_active"] = False
                    self.context.state_data.pop("exp2_yaw_correction_started_at", None)
                    self.context.state_data["exp2_next_axis"] = "center"
                    self.base.stop()
                    settle_seconds = float(settings.get("post_correction_camera_settle_seconds", 0.0))
                    self.context.state_data["exp2_camera_ready_at"] = now + settle_seconds
                    print(
                        "[exp2] continuous yaw segment reached max_seconds={:.3f}; "
                        "switching to center".format(maximum_seconds)
                    )
                    return StepOutcome(
                        observation=observation,
                        reason="exp2 continuous yaw segment time limit",
                    )
                self.context.state_data["exp2_axis_motion_active"] = True
                self._correct_yaw(yaw_error, settings)
                return StepOutcome(observation=observation)

            self.context.state_data.pop("exp2_yaw_correction_started_at", None)
            if bool(self.context.state_data.pop("exp2_axis_motion_active", False)):
                self.context.state_data["stable_frames"] = 0
                self.base.stop()
                self.context.state_data["exp2_next_axis"] = "center"
                settle_seconds = float(settings.get("post_correction_camera_settle_seconds", 0.0))
                self.context.state_data["exp2_camera_ready_at"] = time.monotonic() + settle_seconds
                print("[exp2] yaw accepted; switching to center after {:.3f}s".format(settle_seconds))
                return StepOutcome(observation=observation)
            if center_out:
                self.context.state_data["stable_frames"] = 0
                self.base.stop()
                self.context.state_data["exp2_next_axis"] = "center"
                print("[exp2] yaw accepted; switching to center")
                return StepOutcome(observation=observation)
        else:
            center_limit = float(settings["center_tolerance_norm"])
            if bool(self.context.state_data.get("exp2_center_guard_active", False)):
                center_limit = float(settings["coarse_yaw_center_guard_norm"])
            if abs(center_error) > center_limit:
                self.context.state_data["stable_frames"] = 0
                direction = (
                    settings["center_positive_drive_direction"]
                    if center_error > 0.0
                    else self._opposite(settings["center_positive_drive_direction"])
                )
                self.context.state_data["exp2_axis_motion_active"] = True
                self.base.start_motion(
                    direction,
                    float(settings["drive_speed"]),
                    "exp2_center_align",
                )
                print("[exp2] continuous center correction direction={}".format(direction))
                return StepOutcome(observation=observation)

            self.context.state_data.pop("exp2_center_guard_active", None)
            if bool(self.context.state_data.pop("exp2_axis_motion_active", False)):
                self.context.state_data["stable_frames"] = 0
                self.base.stop()
                self.context.state_data["exp2_next_axis"] = "yaw"
                settle_seconds = float(settings.get("post_correction_camera_settle_seconds", 0.0))
                self.context.state_data["exp2_camera_ready_at"] = time.monotonic() + settle_seconds
                print("[exp2] center accepted; switching to yaw after {:.3f}s".format(settle_seconds))
                return StepOutcome(observation=observation)
            if yaw_out:
                self.context.state_data["stable_frames"] = 0
                self.base.stop()
                self.context.state_data["exp2_next_axis"] = "yaw"
                print("[exp2] center accepted; switching to yaw")
                return StepOutcome(observation=observation)

        self.base.stop()

        stable = int(self.context.state_data.get("stable_frames", 0)) + 1
        self.context.state_data["stable_frames"] = stable
        print("[exp2] stable frame {}/{}".format(stable, int(settings["stable_frames"])))
        if stable >= int(settings["stable_frames"]):
            return StepOutcome(MissionEvent.TARGET_STABLE, observation)
        return StepOutcome(observation=observation)
