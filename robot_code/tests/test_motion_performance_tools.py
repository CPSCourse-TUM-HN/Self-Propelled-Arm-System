import unittest

import ipywidgets as widgets

from tuning_tools.motion_performance_tools import MotionPerformanceUI


class MotionPerformanceUITest(unittest.TestCase):
    def test_command_speeds_are_direct_numeric_inputs(self):
        panel = MotionPerformanceUI()
        for widget in (
            panel.path_speed,
            panel.path_turn_speed,
            panel.manual_speed,
            panel.manual_turn_speed,
            panel.arc_speed,
            panel.arc_turn_speed,
            panel.cruise_fast_speed,
            panel.cruise_slow_speed,
            panel.timed_linear_speed,
            panel.left_turn_speed,
            panel.right_turn_speed,
        ):
            self.assertIsInstance(widget, widgets.FloatText)

    def test_direct_speed_inputs_reject_out_of_range_values(self):
        panel = MotionPerformanceUI()
        panel.manual_turn_speed.value = 1.1
        with self.assertRaises(ValueError):
            panel._validate_motion_inputs()

    def test_angle_turn_uses_measured_direction_specific_time(self):
        panel = MotionPerformanceUI()
        commands = []
        panel._interruptible_pulse = lambda direction, speed, seconds, label: commands.append(
            (direction, float(speed), float(seconds), label)
        )
        panel._turn_angle(3.14159265359, 0.15, "left_half_turn")
        panel._turn_angle(-3.14159265359, 0.3, "right_half_turn")
        self.assertAlmostEqual(commands[0][2], 20.8)
        self.assertAlmostEqual(commands[1][2], 6.8)

    def test_timed_square_defaults_use_measured_fast_turn_times(self):
        panel = MotionPerformanceUI()
        self.assertEqual(panel.left_turn_speed.value, 0.3)
        self.assertEqual(panel.right_turn_speed.value, 0.3)
        self.assertAlmostEqual(panel.left_turn_seconds.value, 3.6)
        self.assertAlmostEqual(panel.right_turn_seconds.value, 3.4)

    def test_timed_square_has_direction_specific_turn_phases(self):
        panel = MotionPerformanceUI()
        panel.timed_linear_speed.value = 0.6
        panel.timed_linear_seconds.value = 1.25
        panel.timed_turn_direction.value = "right"
        phases = panel.timed_square_phases()
        self.assertEqual(len(phases), 8)
        self.assertEqual([phase[0] for phase in phases], ["forward", "right"] * 4)
        self.assertTrue(all(phase[1:3] == (0.6, 1.25) for phase in phases[0::2]))
        for phase in phases[1::2]:
            self.assertAlmostEqual(phase[1], 0.3)
            self.assertAlmostEqual(phase[2], 3.4)


if __name__ == "__main__":
    unittest.main()
