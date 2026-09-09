from __future__ import print_function

import threading
import unittest

from demo_core.robot_control import ArmController, BaseController


class FakeBaseConfig(object):
    def __init__(self, scales=None, dry_run=False):
        self.scales = scales or {"forward": 1.0, "backward": 1.0, "left": 1.0, "right": 1.0}
        self.dry_run = bool(dry_run)

    def get(self, path, default=None):
        if path == "runtime.dry_run.base":
            return self.dry_run
        prefix = "base_motion_command_scales."
        if path.startswith(prefix):
            return self.scales.get(path[len(prefix):], default)
        return default


class FakeRobot(object):
    def __init__(self):
        self.commands = []

    def forward(self, speed): self.commands.append(("forward", speed))
    def backward(self, speed): self.commands.append(("backward", speed))
    def left(self, speed): self.commands.append(("left", speed))
    def right(self, speed): self.commands.append(("right", speed))
    def stop(self): self.commands.append(("stop", None))


class FakeMotionTracker(object):
    def __init__(self):
        self.records = []

    def record_motion(self, direction, speed, seconds):
        self.records.append((direction, speed, seconds))


class BaseCommandScaleTests(unittest.TestCase):
    def controller(self, scales=None, dry_run=False):
        controller = BaseController(FakeBaseConfig(scales, dry_run=dry_run))
        controller.robot = FakeRobot() if not dry_run else None
        return controller

    def test_default_scales_preserve_requested_speed(self):
        controller = self.controller()
        for direction in ("forward", "backward", "left", "right"):
            controller.start_motion(direction, 0.2, direction)
            controller.stop()
            self.assertIn((direction, 0.2), controller.robot.commands)

    def test_command_snapshot_is_read_only_and_clears_on_stop(self):
        controller = self.controller()
        controller.start_motion("forward", 0.2, "hud_test")
        snapshot = controller.command_snapshot()
        snapshot["speed"] = 0.9
        self.assertAlmostEqual(controller.command_snapshot()["speed"], 0.2)
        self.assertEqual(controller.command_snapshot()["label"], "hud_test")
        controller.stop()
        self.assertIsNone(controller.command_snapshot())

    def test_each_direction_uses_its_own_scale(self):
        scales = {"forward": 0.5, "backward": 0.75, "left": 1.2, "right": 0.8}
        controller = self.controller(scales)
        expected = {"forward": 0.1, "backward": 0.15, "left": 0.24, "right": 0.16}
        for direction, speed in expected.items():
            controller.start_motion(direction, 0.2, direction)
            controller.stop()
            actual = [item[1] for item in controller.robot.commands if item[0] == direction][-1]
            self.assertAlmostEqual(actual, speed)

    def test_odometry_records_effective_speed(self):
        controller = self.controller({"forward": 1.25, "backward": 1.0, "left": 1.0, "right": 1.0}, dry_run=True)
        tracker = FakeMotionTracker()
        controller.attach_motion_tracker(tracker)
        controller.pulse("forward", 0.4, 2.0, "scaled")
        self.assertEqual(tracker.records, [("forward", 0.5, 2.0)])

    def test_effective_speed_over_one_is_rejected(self):
        controller = self.controller({"forward": 2.0, "backward": 1.0, "left": 1.0, "right": 1.0})
        with self.assertRaises(ValueError):
            controller.start_motion("forward", 0.6, "unsafe")


class FakeConfig(object):
    def __init__(self):
        self.arm = {
            "speed": 85,
            "servo_settle_seconds": 0.2,
            "poses": {
                "test": {
                    "angles": {"s1": 1, "s2": 2, "s3": 3},
                    "order": [1, 2, 3],
                    "pause_seconds": 0.0,
                }
            },
        }

    def section(self, name):
        if name != "arm":
            raise KeyError(name)
        return self.arm

    def get(self, path, default=None):
        if path == "runtime.dry_run.arm":
            return False
        return default


class FakeTTL(object):
    def __init__(self):
        self.positions = {1: 501, 2: 502, 3: 503, 4: 504, 5: 505}
        self.commands = []
        self.sync_calls = []
        self.first_command = threading.Event()

    def servoAngleCtrl(self, servo_id, angle, direction, speed):
        self.commands.append((servo_id, angle, direction, speed))
        self.first_command.set()
        return 1000 + int(servo_id)

    def nowPosUpdate(self, servo_id):
        return self.positions[int(servo_id)]

    def syncCtrl(self, servo_ids, speeds, positions):
        self.sync_calls.append((list(servo_ids), list(speeds), list(positions)))


class ArmStopTests(unittest.TestCase):
    def controller(self):
        controller = ArmController(FakeConfig())
        controller.ttl = FakeTTL()
        return controller

    def test_stop_and_hold_reissues_current_raw_positions(self):
        controller = self.controller()
        positions = controller.stop_and_hold()
        self.assertEqual(positions, {1: 501, 2: 502, 3: 503, 4: 504, 5: 505})
        self.assertEqual(
            controller.ttl.sync_calls,
            [([1, 2, 3, 4, 5], [85, 85, 85, 85, 85], [501, 502, 503, 504, 505])],
        )

    def test_stop_interrupts_remaining_pose_commands(self):
        controller = self.controller()
        worker = threading.Thread(target=lambda: controller.pose("test"))
        worker.start()
        self.assertTrue(controller.ttl.first_command.wait(1.0))
        controller.stop_and_hold()
        worker.join(1.0)
        self.assertFalse(worker.is_alive())
        self.assertTrue(controller.last_pose_interrupted)
        self.assertEqual([command[0] for command in controller.ttl.commands], [1])

    def test_cooperative_cancel_does_not_issue_competing_servo_commands(self):
        controller = self.controller()
        controller.cancel_motion()
        self.assertTrue(controller.motion_cancelled())
        self.assertEqual(controller.ttl.commands, [])
        self.assertEqual(controller.ttl.sync_calls, [])
        self.assertFalse(controller.wait_for_positions({1: 1001}, "test"))


if __name__ == "__main__":
    unittest.main()
