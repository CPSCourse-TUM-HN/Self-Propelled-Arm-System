import math
import threading
import time


class BaseController(object):
    def __init__(self, config):
        self.config = config
        self.robot = None
        self._motion_lock = threading.RLock()
        self._active_motion = None
        self._blocking_motion = None

        self.motion_tracker = None

    def attach_motion_tracker(self, motion_tracker):
        self.motion_tracker = motion_tracker

    def command_snapshot(self):
        """Return the currently issued command without exposing mutable controller state."""
        with self._motion_lock:
            motion = self._blocking_motion or self._active_motion
            return dict(motion) if motion is not None else None

    def _begin_blocking_motion(self, direction, requested_speed, effective_speed, label):
        with self._motion_lock:
            self._blocking_motion = {
                "direction": direction,
                "speed": effective_speed,
                "requested_speed": requested_speed,
                "label": label,
                "started_at": time.time(),
            }

    def _finish_blocking_motion(self):
        with self._motion_lock:
            self._blocking_motion = None

    def _effective_speed(self, direction, requested_speed):
        requested_speed = float(requested_speed)
        scale = float(self.config.get("base_motion_command_scales.{}".format(direction), 1.0))
        effective_speed = requested_speed * scale
        if not math.isfinite(requested_speed) or requested_speed < 0.0:
            raise ValueError("base speed must be non-negative")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("base motion command scale must be positive for {}".format(direction))
        if effective_speed > 1.0:
            raise ValueError(
                "effective base speed exceeds 1.0: direction={} requested={} scale={} effective={}".format(
                    direction, requested_speed, scale, effective_speed
                )
            )
        return effective_speed

    def _record_motion(self, direction, speed, effective_seconds):
        if self.motion_tracker is not None and float(effective_seconds) > 0.0:
            self.motion_tracker.record_motion(direction, speed, effective_seconds)

    def connect(self):
        if self.config.get("runtime.dry_run.base", True):
            print("[base] dry_run=True")
            return None
        if self.robot is None:
            from jetbot import Robot

            self.robot = Robot()
            print("[base] Robot connected")
        return self.robot

    def stop(self):
        with self._motion_lock:
            if self.robot is not None:
                self.robot.stop()
            if self._active_motion is not None:
                motion = self._active_motion
                elapsed = max(0.0, time.time() - float(motion["started_at"]))
                self._record_motion(motion["direction"], motion["speed"], elapsed)
                print("[base] continuous stop label={} elapsed={:.3f}s".format(motion["label"], elapsed))
                self._active_motion = None
            self._blocking_motion = None
        print("[base] stop")

    def start_motion(self, direction, speed, label):
        """Start or retain a non-blocking constant-speed command until stop() is called."""
        requested_speed = float(speed)
        speed = self._effective_speed(direction, requested_speed)
        with self._motion_lock:
            if self._active_motion is not None:
                current = self._active_motion
                if current["direction"] == direction and abs(float(current["speed"]) - speed) < 1e-9:
                    return False
                self.stop()
            bot = self.connect()
            if bot is not None:
                if direction == "forward":
                    bot.forward(speed)
                elif direction == "backward":
                    bot.backward(speed)
                elif direction == "left":
                    bot.left(speed)
                elif direction == "right":
                    bot.right(speed)
                else:
                    raise ValueError("unknown base direction: {}".format(direction))
            self._active_motion = {
                "direction": direction,
                "speed": speed,
                "requested_speed": requested_speed,
                "label": label,
                "started_at": time.time(),
            }
            print(
                "[base] continuous start label={} direction={} requested_speed={} effective_speed={}".format(
                    label, direction, requested_speed, speed
                )
            )
            return True

    def motion_active(self, label=None):
        with self._motion_lock:
            if self._active_motion is None:
                return False
            return label is None or self._active_motion["label"] == label

    def update_motion_odometry(self, dry_run_step_seconds=0.0):
        """Record elapsed continuous motion without stopping the motors."""
        with self._motion_lock:
            if self._active_motion is None:
                return 0.0
            now = time.time()
            if self.config.get("runtime.dry_run.base", True) and float(dry_run_step_seconds) > 0.0:
                elapsed = float(dry_run_step_seconds)
            else:
                elapsed = max(0.0, now - float(self._active_motion["started_at"]))
            self._record_motion(
                self._active_motion["direction"],
                self._active_motion["speed"],
                elapsed,
            )
            self._active_motion["started_at"] = now
            return elapsed

    def pulse(self, direction, speed, seconds, label):
        requested_speed = float(speed)
        speed = self._effective_speed(direction, requested_speed)
        bot = self.connect()
        print(
            "[base] {} direction={} requested_speed={} effective_speed={} seconds={}".format(
                label, direction, requested_speed, speed, seconds
            )
        )
        if bot is None:
            self._record_motion(direction, speed, seconds)
            return

        start = time.time()
        self._begin_blocking_motion(direction, requested_speed, speed, label)
        try:
            if direction == "forward":
                bot.forward(float(speed))
            elif direction == "backward":
                bot.backward(float(speed))
            elif direction == "left":
                bot.left(float(speed))
            elif direction == "right":
                bot.right(float(speed))
            else:
                raise ValueError("unknown base direction: {}".format(direction))
            time.sleep(float(seconds))
        finally:
            bot.stop()
            self._finish_blocking_motion()
            self._record_motion(direction, speed, min(float(seconds), time.time() - start))

class ArmController(object):
    def __init__(self, config):
        self.config = config
        self.settings = config.section("arm")
        self.ttl = None
        self._motion_cancelled = threading.Event()
        self._io_lock = threading.Lock()
        self.last_pose_interrupted = False

    def connect(self):
        if self.config.get("runtime.dry_run.arm", True):
            print("[arm] dry_run=True")
            return None
        if self.ttl is None:
            from SCSCtrl import TTLServo

            self.ttl = TTLServo
            print("[arm] TTLServo connected")
        return self.ttl

    def move_servo(self, servo_id, angle, speed, label):
        ttl = self.connect()
        print("[arm] {} servo={} angle={} speed={}".format(label, servo_id, angle, speed))
        target_position = None
        if ttl is not None:
            with self._io_lock:
                target_position = ttl.servoAngleCtrl(int(servo_id), int(angle), 1, int(speed))
            self._motion_cancelled.wait(float(self.settings.get("servo_settle_seconds", 0.35)))
        return target_position

    def stop_and_hold(self):
        """Cancel the active pose and command every servo to hold its current raw position."""
        self._motion_cancelled.set()
        self.last_pose_interrupted = True
        ttl = self.connect()
        if ttl is None:
            print("[arm] stop_and_hold dry_run=True")
            return {}

        servo_ids = [1, 2, 3, 4, 5]
        with self._io_lock:
            positions = []
            for servo_id in servo_ids:
                positions.append(int(ttl.nowPosUpdate(servo_id)))
            speeds = [int(self.settings.get("speed", 85))] * len(servo_ids)
            if not hasattr(ttl, "syncCtrl"):
                raise RuntimeError("TTLServo.syncCtrl is unavailable; cannot hold current arm position")
            ttl.syncCtrl(servo_ids, speeds, positions)
        result = dict(zip(servo_ids, positions))
        print("[arm] stop_and_hold positions={}".format(result))
        return result

    def cancel_motion(self):
        """Request a cooperative stop without issuing competing servo commands."""
        self._motion_cancelled.set()
        self.last_pose_interrupted = True
        print("[arm] cancellation requested")

    def motion_cancelled(self):
        return self._motion_cancelled.is_set()

    def pose(self, name, pose=None):
        self._motion_cancelled.clear()
        self.last_pose_interrupted = False
        definition = self.settings["poses"][name]
        pose = pose or definition["angles"]
        speed = int(self.settings["speed"])
        order = definition.get("order", [5, 4, 3, 2, 1])
        target_positions = {}
        print("[arm] pose {}".format(name))
        for sid in order:
            if self._motion_cancelled.is_set():
                self.last_pose_interrupted = True
                break
            key = "s{}".format(sid)
            if key in pose:
                target = self.move_servo(sid, pose[key], speed, name)
                if target is not None:
                    target_positions[int(sid)] = int(target)
            if self._motion_cancelled.is_set():
                self.last_pose_interrupted = True
                break
        if self.last_pose_interrupted:
            print("[arm] pose {} interrupted; completed_servos={}".format(name, sorted(target_positions)))
            return target_positions
        if self.ttl is not None:
            time.sleep(float(definition.get("pause_seconds", 0.0)))
        return target_positions

    def wait_for_positions(self, target_positions, label):
        if self.config.get("runtime.dry_run.arm", True):
            print("[arm-position] {} dry_run=True; position wait skipped".format(label))
            return True
        wait = self.settings.get("position_wait", {})
        if not bool(wait.get("enabled", True)):
            print("[arm-position] {} disabled; position wait skipped".format(label))
            return True
        if not target_positions:
            print("[arm-position] {} has no raw targets".format(label))
            return False

        ttl = self.connect()
        stability_delta = int(wait.get("stability_delta_raw", 3))
        timeout_seconds = float(wait.get("timeout_seconds", 10.0))
        min_wait_seconds = float(wait.get("min_wait_seconds", 0.5))
        poll_seconds = float(wait.get("poll_seconds", 0.15))
        stable_required = max(1, int(wait.get("stable_samples", 2)))
        stable_samples = 0
        start = time.time()
        last_positions = {}
        previous_positions = None

        print(
            "[arm-position] {} waiting for motion to settle servos={} stability_delta={} min_wait={} timeout={} stable_samples={}".format(
                label,
                sorted(target_positions),
                stability_delta,
                min_wait_seconds,
                timeout_seconds,
                stable_required,
            )
        )
        while time.time() - start < timeout_seconds:
            if self._motion_cancelled.is_set():
                print("[arm-position] {} cancelled".format(label))
                return False
            read_errors = []
            last_positions = {}
            for servo_id in sorted(target_positions):
                try:
                    with self._io_lock:
                        current = int(ttl.nowPosUpdate(int(servo_id)))
                    last_positions[int(servo_id)] = current
                except Exception as exc:
                    read_errors.append(int(servo_id))
                    last_positions[int(servo_id)] = "error:{}".format(exc)

            movement = {}
            motion_stable = previous_positions is not None and not read_errors
            if previous_positions is not None:
                for servo_id, current in last_positions.items():
                    previous = previous_positions.get(servo_id)
                    if not isinstance(current, int) or not isinstance(previous, int):
                        motion_stable = False
                        continue
                    delta = abs(current - previous)
                    movement[servo_id] = delta
                    if delta > stability_delta:
                        motion_stable = False

            elapsed = time.time() - start
            if motion_stable and elapsed >= min_wait_seconds:
                stable_samples += 1
                if stable_samples >= stable_required:
                    print(
                        "[arm-position] {} settled positions={} movement={} elapsed={:.2f}s".format(
                            label,
                            last_positions,
                            movement,
                            time.time() - start,
                        )
                    )
                    return True
            else:
                stable_samples = 0
                print(
                    "[arm-position] {} waiting positions={} movement={} read_errors={}".format(
                        label,
                        last_positions,
                        movement,
                        read_errors,
                    )
                )
            previous_positions = dict(last_positions)
            time.sleep(max(0.02, poll_seconds))

        print(
            "[arm-position] {} motion did not settle before timeout positions={}; base push blocked".format(
                label,
                last_positions,
            )
        )
        return False
