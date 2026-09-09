import math
import unittest

from demo_core import load_config
from demo_core.robot_control import BaseController
from demo_core.vague_map import (
    CommandOdometry,
    Point2D,
    Pose2D,
    VagueMap,
    VagueMapNavigator,
    linear_distance_meters,
    normalize_heading,
    point_in_robot_frame,
    robot_pose_from_tag_pose,
)


class FakeBase(object):
    def __init__(self, tracker=None):
        self.tracker = tracker
        self.commands = []
        self.active_motion = None

    def pulse(self, direction, speed, seconds, label):
        self.commands.append((direction, float(speed), float(seconds), label))
        if self.tracker is not None:
            self.tracker.record_motion(direction, speed, seconds)

    def start_motion(self, direction, speed, label):
        self.active_motion = (direction, float(speed), label)
        self.commands.append((direction, float(speed), "continuous", label))

    def update_motion_odometry(self, dry_run_step_seconds=0.0):
        return 0.0

    def motion_active(self, label=None):
        return self.active_motion is not None and (label is None or self.active_motion[2] == label)

    def stop(self):
        self.active_motion = None
        self.commands.append(("stop",))


class VagueMapGeometryTest(unittest.TestCase):
    def setUp(self):
        self.settings = load_config().section("vague_map")
        self.vague_map = VagueMap(self.settings)
        self.odometry = CommandOdometry(self.vague_map, self.settings["odometry"])

    def test_forward_backward_and_turn_signs(self):
        self.odometry.record_motion("forward", 0.5, 1.0)
        forward_distance = linear_distance_meters(self.settings["odometry"], 0.5, 1.0)
        self.assertAlmostEqual(self.vague_map.robot_pose.y_m, forward_distance)
        self.odometry.record_motion("backward", 0.5, 0.5)
        self.assertAlmostEqual(self.vague_map.robot_pose.y_m, forward_distance / 2.0)
        self.odometry.record_motion("left", 0.5, 1.0)
        self.assertLess(self.vague_map.robot_pose.heading_rad, 0.0)
        self.odometry.record_motion("right", 0.5, 1.0)
        self.assertAlmostEqual(self.vague_map.robot_pose.heading_rad, 0.0)

    def test_heading_normalization(self):
        self.assertAlmostEqual(normalize_heading(3.0 * math.pi), -math.pi)
        self.odometry.record_motion("left", 1.0, 3.0)
        self.assertGreaterEqual(self.vague_map.robot_pose.heading_rad, -math.pi)
        self.assertLess(self.vague_map.robot_pose.heading_rad, math.pi)

    def test_calibrated_turn_odometry_uses_direction_and_speed_sample(self):
        config = load_config()
        odometry = CommandOdometry(
            self.vague_map,
            self.settings["odometry"],
            config.section("base_turn_response"),
        )
        odometry.record_motion("left", 0.15, 10.4)
        self.assertAlmostEqual(self.vague_map.robot_pose.heading_rad, -math.pi / 2.0, places=7)
        odometry.record_motion("right", 0.15, 9.0)
        self.assertAlmostEqual(self.vague_map.robot_pose.heading_rad, 0.0, places=7)

    def test_linear_odometry_uses_exact_samples_and_interpolates_between_them(self):
        odometry = self.settings["odometry"]
        self.assertAlmostEqual(linear_distance_meters(odometry, 0.2, 4.0), 0.13)
        self.assertAlmostEqual(linear_distance_meters(odometry, 0.6, 4.0), 0.415)
        lower = 0.13 / (0.2 * 4.0)
        upper = 0.415 / (0.6 * 4.0)
        expected = 0.4 * 4.0 * (lower + upper) / 2.0
        self.assertAlmostEqual(linear_distance_meters(odometry, 0.4, 4.0), expected)

    def test_image_sides_map_to_world_x(self):
        observation = {"found": True, "center_y": 180.0, "error_x": -0.5}
        point = self.vague_map.estimate_can_position(observation, 0.5)
        self.assertLess(point.x_m, 0.0)
        observation["error_x"] = 0.5
        point = self.vague_map.estimate_can_position(observation, 0.5)
        self.assertGreater(point.x_m, 0.0)

    def test_map_heading_correction_uses_continuous_slow_turn(self):
        self.vague_map.robot_pose.heading_rad = math.pi / 2.0
        base = FakeBase(self.odometry)
        navigator = VagueMapNavigator(self.vague_map, base, self.settings["navigation"])
        navigator.step_toward(Point2D(0.0, 0.5), "test_target")
        self.assertEqual(base.commands[-1], ("left", 0.15, "continuous", "map_turn_test_target"))

    def test_robot_pose_is_recovered_from_tag_pose_and_camera_mount(self):
        pose = robot_pose_from_tag_pose(
            {"x_m": 0.0, "y_m": 0.0, "heading_rad": 0.0},
            {
                "x": -0.1,
                "y": 0.0,
                "z": 0.9,
                "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            },
            {"x_m": 0.0, "y_m": 0.0, "heading_rad": 0.0},
        )
        self.assertAlmostEqual(pose.x_m, 0.1)
        self.assertAlmostEqual(pose.y_m, -0.9)
        self.assertAlmostEqual(pose.heading_rad, 0.0)

        side_pose = robot_pose_from_tag_pose(
            {"x_m": 0.0, "y_m": 0.0, "heading_rad": 0.0},
            {
                "x": 0.0,
                "y": 0.0,
                "z": 0.9,
                "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            },
            {"x_m": 0.0, "y_m": 0.0, "heading_rad": math.pi / 2.0},
        )
        self.assertAlmostEqual(side_pose.heading_rad, -math.pi / 2.0)

    def test_right_side_bin_is_positive_robot_x(self):
        relative = point_in_robot_frame(
            {"x_m": 0.75, "y_m": 0.0},
            {"x_m": 0.5, "y_m": 0.0, "heading_rad": 0.0},
        )
        self.assertAlmostEqual(relative.x_m, 0.25)
        self.assertAlmostEqual(relative.y_m, 0.0)


class VagueMapMemoryTest(unittest.TestCase):
    def setUp(self):
        self.settings = load_config().section("vague_map")
        self.vague_map = VagueMap(self.settings)

    def test_bounds_distinct_candidates_nearest_and_remove(self):
        outside_x = float(self.settings["bounds_m"]["max_x"]) + 1.0
        self.assertIsNone(self.vague_map.remember_can(Point2D(outside_x, 0.0), 0.9))
        first = self.vague_map.remember_can(Point2D(0.4, 0.1), 0.7)
        merged = self.vague_map.remember_can(Point2D(0.5, 0.1), 0.8)
        second = self.vague_map.remember_can(Point2D(-0.7, 0.0), 0.9)
        self.assertNotEqual(first.can_id, merged.can_id)
        self.assertEqual(len(self.vague_map.known_cans), 3)
        self.assertEqual(self.vague_map.nearest_can().can_id, first.can_id)
        self.vague_map.remove_can(second.can_id, "test")
        self.assertNotIn(second.can_id, self.vague_map.known_cans)

    def test_pose_correction_transforms_all_remembered_cans(self):
        first = self.vague_map.remember_can(Point2D(0.2, 0.5), 0.8)
        second = self.vague_map.remember_can(Point2D(-0.3, 0.4), 0.7)
        before = [
            point_in_robot_frame(item.position, self.vague_map.robot_pose).as_dict()
            for item in (first, second)
        ]
        self.vague_map.correct_robot_pose(Pose2D(0.5, -0.2, math.pi / 2.0), "test")
        after = [
            point_in_robot_frame(item.position, self.vague_map.robot_pose).as_dict()
            for item in (first, second)
        ]
        for expected, actual in zip(before, after):
            self.assertAlmostEqual(expected["x_m"], actual["x_m"])
            self.assertAlmostEqual(expected["y_m"], actual["y_m"])

    def test_bin_docking_resets_pose(self):
        self.vague_map.robot_pose = Pose2D(0.7, 0.7, 0.2)
        self.vague_map.set_robot_pose(self.vague_map.bin_docking_pose, "test")
        self.assertEqual(self.vague_map.robot_pose.as_dict(), self.vague_map.bin_docking_pose.as_dict())


if __name__ == "__main__":
    unittest.main()
