from __future__ import print_function

import math
import threading
import time
import traceback

import ipywidgets as widgets
from IPython.display import display

from demo_core import CommandOdometry, TurnResponseModel, VagueMap, load_config
from demo_core.robot_control import BaseController
from demo_core.vague_map import normalize_heading


class MotionPerformanceUI(object):
    """Board-side base calibration and trajectory test panel."""

    def __init__(self):
        config = load_config()
        vague = config.section("vague_map")
        odometry = vague["odometry"]
        navigation = vague["navigation"]
        speed_profiles = config.section("base_motion_speed_profiles")
        self.turn_response_settings = config.section("base_turn_response")
        self.output = widgets.Output(layout={"border": "1px solid #ccc", "height": "520px", "overflow_y": "auto"})
        self.base_real = widgets.Checkbox(value=False, description="base_real")
        self.linear_scale = widgets.FloatText(
            value=float(odometry["linear_meters_per_speed_second"]), description="linear_m/(v*s)"
        )
        self.slip = widgets.FloatText(value=float(odometry.get("linear_slip_factor", 1.0)), description="linear_slip")
        self.angular_scale = widgets.FloatText(
            value=float(odometry["angular_radians_per_speed_second"]), description="angular_rad/(v*s)"
        )
        self.chunk_seconds = widgets.FloatText(value=0.10, description="pulse_chunk_s")

        self.square_x = widgets.FloatText(value=0.4, description="point_x_m")
        self.square_y = widgets.FloatText(value=0.0, description="point_y_m")
        self.square_loops = widgets.IntText(value=1, description="loops")
        self.path_speed = widgets.FloatText(value=float(navigation["forward_speed"]), description="path_speed")
        self.path_turn_speed = widgets.FloatText(value=float(navigation["turn_speed"]), description="path_turn_speed")

        self.manual_speed = widgets.FloatText(value=float(speed_profiles["linear"]["slow"]), description="manual_speed")
        self.manual_seconds = widgets.FloatText(value=0.5, description="manual_seconds")
        self.manual_turn_speed = widgets.FloatText(value=float(speed_profiles["turn"]["slow"]), description="turn_speed")
        self.manual_turn_seconds = widgets.FloatText(value=0.5, description="turn_seconds")

        timed_turn_speed = float(speed_profiles["turn"]["fast"])
        timed_turn_model = TurnResponseModel(
            self.turn_response_settings,
            odometry["angular_radians_per_speed_second"],
        )
        self.timed_turn_direction = widgets.Dropdown(
            options=(("Left / CCW", "left"), ("Right / CW", "right")),
            value="left",
            description="square_turn",
        )
        self.timed_linear_speed = widgets.FloatText(
            value=float(speed_profiles["linear"]["slow"]), description="linear_speed"
        )
        self.timed_linear_seconds = widgets.FloatText(value=1.0, description="linear_seconds")
        self.left_turn_speed = widgets.FloatText(value=timed_turn_speed, description="left_speed")
        self.left_turn_seconds = widgets.FloatText(
            value=timed_turn_model.seconds_for_angle(math.pi / 2.0, "left", timed_turn_speed),
            description="left_90_seconds",
        )
        self.right_turn_speed = widgets.FloatText(value=timed_turn_speed, description="right_speed")
        self.right_turn_seconds = widgets.FloatText(
            value=timed_turn_model.seconds_for_angle(math.pi / 2.0, "right", timed_turn_speed),
            description="right_90_seconds",
        )

        self.arc_radius = widgets.FloatText(value=0.4, description="arc_radius_m")
        self.arc_speed = widgets.FloatText(value=float(speed_profiles["linear"]["slow"]), description="arc_speed")
        self.arc_turn_speed = widgets.FloatText(value=float(speed_profiles["turn"]["slow"]), description="arc_turn_speed")
        self.arc_segments = widgets.IntText(value=12, description="arc_segments")
        self.arc_direction = widgets.Dropdown(options=("left", "right"), value="left", description="arc_direction")

        self.cruise_fast_speed = widgets.FloatText(value=float(speed_profiles["linear"]["fast"]), description="fast_speed")
        self.cruise_slow_speed = widgets.FloatText(value=float(speed_profiles["linear"]["slow"]), description="slow_speed")
        self.cruise_fast_seconds = widgets.FloatText(value=1.0, description="fast_phase_s")
        self.cruise_slow_seconds = widgets.FloatText(value=1.0, description="slow_phase_s")

        self.stop_event = threading.Event()
        self.worker = None
        self.base = None
        self.vague_map = None
        self._worker_lock = threading.Lock()

    def _new_session(self):
        config = load_config(overrides={"runtime": {"dry_run": {"base": not bool(self.base_real.value)}}})
        settings = dict(config.section("vague_map"))
        settings["odometry"] = dict(settings["odometry"])
        settings["odometry"]["linear_response_samples"] = []
        settings["odometry"].update({
            "linear_meters_per_speed_second": float(self.linear_scale.value),
            "linear_slip_factor": float(self.slip.value),
            "angular_radians_per_speed_second": float(self.angular_scale.value),
        })
        self.vague_map = VagueMap(settings)
        self.base = BaseController(config)
        self.base.attach_motion_tracker(CommandOdometry(
            self.vague_map,
            settings["odometry"],
            config.section("base_turn_response"),
        ))
        self.stop_event.clear()
        return self.base

    def _validate_motion_inputs(self):
        for name, value in (
            ("linear_m/(v*s)", self.linear_scale.value),
            ("linear_slip", self.slip.value),
            ("angular_rad/(v*s)", self.angular_scale.value),
            ("pulse_chunk_s", self.chunk_seconds.value),
        ):
            if float(value) <= 0.0:
                raise ValueError("{} must be positive".format(name))
        for name, widget in (
            ("path_speed", self.path_speed),
            ("path_turn_speed", self.path_turn_speed),
            ("manual_speed", self.manual_speed),
            ("turn_speed", self.manual_turn_speed),
            ("arc_speed", self.arc_speed),
            ("arc_turn_speed", self.arc_turn_speed),
            ("fast_speed", self.cruise_fast_speed),
            ("slow_speed", self.cruise_slow_speed),
            ("timed_linear_speed", self.timed_linear_speed),
            ("left_speed", self.left_turn_speed),
            ("right_speed", self.right_turn_speed),
        ):
            value = float(widget.value)
            if value <= 0.0 or value > 1.0:
                raise ValueError("{} must be in (0.0, 1.0]".format(name))
        for name, widget in (
            ("timed_linear_seconds", self.timed_linear_seconds),
            ("left_90_seconds", self.left_turn_seconds),
            ("right_90_seconds", self.right_turn_seconds),
        ):
            if float(widget.value) <= 0.0:
                raise ValueError("{} must be positive".format(name))

    def _interruptible_pulse(self, direction, speed, seconds, label):
        speed = float(speed)
        seconds = max(0.0, float(seconds))
        if speed <= 0.0:
            raise ValueError("speed must be positive")
        bot = self.base.connect()
        print("[motion] {} direction={} speed={} seconds={}".format(label, direction, speed, seconds))
        if bot is None:
            if self.stop_event.is_set():
                raise RuntimeError("motion stopped by user")
            self.base._record_motion(direction, speed, seconds)
            return
        started = time.time()
        try:
            self._drive_bot(bot, direction, speed)
            self._wait_until(started + seconds)
        finally:
            bot.stop()
            self.base._record_motion(direction, speed, min(seconds, time.time() - started))

    def _drive_bot(self, bot, direction, speed):
        if direction == "forward":
            bot.forward(float(speed))
        elif direction == "backward":
            bot.backward(float(speed))
        elif direction == "left":
            bot.left(float(speed))
        elif direction == "right":
            bot.right(float(speed))
        else:
            raise ValueError("unknown direction: {}".format(direction))

    def _wait_until(self, deadline):
        while time.time() < deadline:
            if self.stop_event.is_set():
                raise RuntimeError("motion stopped by user")
            time.sleep(min(max(0.02, float(self.chunk_seconds.value)), max(0.0, deadline - time.time())))

    def _continuous_sequence(self, phases, label):
        bot = self.base.connect()
        if bot is None:
            for direction, speed, seconds, phase_label in phases:
                if self.stop_event.is_set():
                    raise RuntimeError("motion stopped by user")
                print("[motion] {} phase={} speed={} seconds={}".format(label, phase_label, speed, seconds))
                self.base._record_motion(direction, speed, seconds)
            return
        try:
            for direction, speed, seconds, phase_label in phases:
                if float(speed) <= 0.0 or float(seconds) <= 0.0:
                    raise ValueError("cruise speed and phase duration must be positive")
                print("[motion] {} phase={} speed={} seconds={}".format(label, phase_label, speed, seconds))
                started = time.time()
                self._drive_bot(bot, direction, speed)
                try:
                    self._wait_until(started + float(seconds))
                finally:
                    self.base._record_motion(direction, speed, min(float(seconds), time.time() - started))
        finally:
            bot.stop()

    def _turn_angle(self, angle_rad, speed, label):
        if abs(float(angle_rad)) < 1e-6:
            return
        direction = "left" if angle_rad > 0.0 else "right"
        model = TurnResponseModel(self.turn_response_settings, self.angular_scale.value)
        seconds = model.seconds_for_angle(angle_rad, direction, speed)
        self._interruptible_pulse(direction, speed, seconds, label)

    def _drive_distance(self, distance_m, speed, label):
        if abs(float(distance_m)) < 1e-6:
            return
        denominator = float(speed) * float(self.linear_scale.value) * float(self.slip.value)
        seconds = abs(float(distance_m)) / denominator
        self._interruptible_pulse("forward" if distance_m > 0.0 else "backward", speed, seconds, label)

    def _go_to(self, x_m, y_m, speed, turn_speed, label):
        pose = self.vague_map.robot_pose
        dx = float(x_m) - pose.x_m
        dy = float(y_m) - pose.y_m
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            return
        heading = math.atan2(dy, dx)
        self._turn_angle(normalize_heading(heading - pose.heading_rad), turn_speed, label + "_turn")
        self._drive_distance(distance, speed, label + "_forward")

    def _launch(self, name, operation):
        with self._worker_lock:
            if self.worker is not None and self.worker.is_alive():
                with self.output:
                    print("[motion] another job is still running")
                return
            self._validate_motion_inputs()
            self._new_session()

            def target():
                started = time.time()
                with self.output:
                    print("\n>>> {} base_real={}".format(name, self.base_real.value))
                    try:
                        operation()
                        print("[motion] COMPLETE {} elapsed={:.2f}s pose={}".format(
                            name, time.time() - started, self.vague_map.robot_pose.as_dict()
                        ))
                    except Exception as exc:
                        print("[motion] STOPPED/FAILED {}: {}".format(name, exc))
                        traceback.print_exc()
                    finally:
                        if self.base is not None:
                            self.base.stop()

            self.worker = threading.Thread(target=target)
            self.worker.daemon = True
            self.worker.start()

    def run_square(self):
        x_m = float(self.square_x.value)
        y_m = float(self.square_y.value)
        loops = int(self.square_loops.value)
        if math.hypot(x_m, y_m) <= 0.0:
            raise ValueError("point_x_m and point_y_m cannot both be zero")
        if loops <= 0:
            raise ValueError("loops must be positive")
        points = ((x_m, y_m), (x_m - y_m, y_m + x_m), (-y_m, x_m), (0.0, 0.0))

        def operation():
            for loop_index in range(loops):
                for point_index, point in enumerate(points):
                    self._go_to(
                        point[0], point[1], float(self.path_speed.value), float(self.path_turn_speed.value),
                        "square_{}_{}".format(loop_index + 1, point_index + 1),
                    )
                print("[motion] square loop {}/{} complete".format(loop_index + 1, loops))

        self._launch("square", operation)

    def run_manual(self, direction):
        def operation():
            turning = direction in ("left", "right")
            speed = self.manual_turn_speed.value if turning else self.manual_speed.value
            seconds = self.manual_turn_seconds.value if turning else self.manual_seconds.value
            self._interruptible_pulse(direction, speed, seconds, "manual_{}".format(direction))

        self._launch("manual_{}".format(direction), operation)

    def timed_square_phases(self):
        direction = str(self.timed_turn_direction.value)
        if direction == "left":
            turn_speed = float(self.left_turn_speed.value)
            turn_seconds = float(self.left_turn_seconds.value)
        else:
            turn_speed = float(self.right_turn_speed.value)
            turn_seconds = float(self.right_turn_seconds.value)
        phases = []
        for side in range(1, 5):
            phases.append((
                "forward", float(self.timed_linear_speed.value),
                float(self.timed_linear_seconds.value), "side_{}".format(side),
            ))
            phases.append((direction, turn_speed, turn_seconds, "corner_{}".format(side)))
        return phases

    def run_timed_square(self):
        self._launch(
            "timed_square",
            lambda: self._continuous_sequence(self.timed_square_phases(), "timed_square"),
        )

    def run_arc(self):
        radius = float(self.arc_radius.value)
        segments = int(self.arc_segments.value)
        if radius <= 0.0 or segments < 2:
            raise ValueError("arc radius must be positive and segments must be at least 2")
        sign = 1.0 if self.arc_direction.value == "left" else -1.0
        delta = sign * math.pi / float(segments)
        chord = 2.0 * radius * math.sin(abs(delta) / 2.0)

        def operation():
            for index in range(segments):
                self._turn_angle(delta / 2.0, float(self.arc_turn_speed.value), "arc_{}_turn_in".format(index + 1))
                self._drive_distance(chord, float(self.arc_speed.value), "arc_{}_forward".format(index + 1))
                self._turn_angle(delta / 2.0, float(self.arc_turn_speed.value), "arc_{}_turn_out".format(index + 1))

        self._launch("semicircle", operation)

    def run_cruise(self):
        def operation():
            self._continuous_sequence([
                ("forward", float(self.cruise_fast_speed.value), float(self.cruise_fast_seconds.value), "fast_1"),
                ("forward", float(self.cruise_slow_speed.value), float(self.cruise_slow_seconds.value), "slow"),
                ("forward", float(self.cruise_fast_speed.value), float(self.cruise_fast_seconds.value), "fast_2"),
            ], "fast_slow_fast")

        self._launch("fast_slow_fast", operation)

    def stop_base(self):
        self.stop_event.set()
        if self.base is not None:
            self.base.stop()
        with self.output:
            print("[motion] STOP requested")

    def clear_log(self):
        self.output.clear_output()

    def _button(self, label, callback, style=""):
        button = widgets.Button(description=label, button_style=style, layout=widgets.Layout(width="145px"))
        button.on_click(lambda _: callback())
        return button

    def display(self):
        square = self._button("Run Square", self.run_square, "success")
        arc = self._button("Run Semicircle", self.run_arc, "success")
        cruise = self._button("Run Fast-Slow-Fast", self.run_cruise, "success")
        timed_square = self._button("Run Timed Square", self.run_timed_square, "success")
        stop = self._button("STOP BASE", self.stop_base, "danger")
        clear = self._button("Clear Log", self.clear_log)
        manual = [self._button(name.title(), lambda value=name: self.run_manual(value)) for name in ("forward", "backward", "left", "right")]

        display(widgets.VBox([
            widgets.HTML("<b>Shared calibration</b>"),
            widgets.HBox([self.base_real, self.linear_scale, self.slip, self.angular_scale, self.chunk_seconds]),
            widgets.HTML("<b>1. Square: origin and (x,y) are adjacent corners</b>"),
            widgets.HBox([self.square_x, self.square_y, self.square_loops, self.path_speed, self.path_turn_speed, square]),
            widgets.HTML("<b>2. Manual speed and turn test</b>"),
            widgets.HBox([self.manual_speed, self.manual_seconds, self.manual_turn_speed, self.manual_turn_seconds]),
            widgets.HBox(manual),
            widgets.HTML("<b>3. Timed square (direct speed × time; in memory only)</b>"),
            widgets.HBox([self.timed_turn_direction, self.timed_linear_speed, self.timed_linear_seconds, timed_square]),
            widgets.HBox([self.left_turn_speed, self.left_turn_seconds, self.right_turn_speed, self.right_turn_seconds]),
            widgets.HTML("<b>4. Segmented semicircle</b>"),
            widgets.HBox([self.arc_radius, self.arc_speed, self.arc_turn_speed, self.arc_segments, self.arc_direction, arc]),
            widgets.HTML("<b>5. Constant-command fast / slow / fast cruise</b>"),
            widgets.HBox([self.cruise_fast_speed, self.cruise_slow_speed, self.cruise_fast_seconds, self.cruise_slow_seconds, cruise]),
            widgets.HBox([stop, clear]),
            self.output,
        ]))
        print("[motion] ready; base_real is off by default")


def build_ui():
    panel = MotionPerformanceUI()
    panel.display()
    return panel
