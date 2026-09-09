from __future__ import print_function

import math
import unittest

from demo_core import SearchingRoutineExecutor, SearchingRoutineLibrary, load_config
from demo_core.vague_map import CommandOdometry, Pose2D, VagueMap


class FakeBase(object):
    def __init__(self):
        self.active = None
        self.starts = []
        self.stop_count = 0

    def start_motion(self, direction, speed, label):
        command = {
            "direction": direction,
            "speed": float(speed),
            "requested_speed": float(speed),
            "label": label,
        }
        if self.active == command:
            return False
        self.active = command
        self.starts.append(dict(command))
        return True

    def motion_active(self, label=None):
        return self.active is not None and (label is None or self.active["label"] == label)

    def command_snapshot(self):
        return dict(self.active) if self.active is not None else None

    def update_motion_odometry(self, dry_run_step_seconds=0.0):
        return float(dry_run_step_seconds)

    def stop(self):
        self.active = None
        self.stop_count += 1


class TrackedFakeBase(FakeBase):
    def __init__(self, tracker):
        super(TrackedFakeBase, self).__init__()
        self.tracker = tracker

    def update_motion_odometry(self, dry_run_step_seconds=0.0):
        elapsed = float(dry_run_step_seconds)
        if self.active is not None:
            self.tracker.record_motion(self.active["direction"], self.active["speed"], elapsed)
        return elapsed


class SearchingRoutineTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.library = SearchingRoutineLibrary(self.config.section("predefined_routines"), self.config)

    def test_requested_routines_load_by_id_and_name(self):
        self.assertEqual(self.library.get(0).name, "rotate_in_place")
        self.assertEqual(self.library.get(2).id, 2)
        self.assertEqual(self.library.get("square_ccw_with_midpoint_spins").id, 3)
        arrival_scan = self.library.get("map_bin_arrival_scan")
        self.assertEqual(arrival_scan.id, 4)
        self.assertAlmostEqual(arrival_scan.steps[0]["angle_rad"], math.radians(400.0))
        self.assertEqual(arrival_scan.steps[0]["speed"], "turn.fast")
        spins = [
            step for step in self.library.get(3).steps
            if step.get("angle_rad", 0.0) > 6.0
        ]
        self.assertEqual(len(spins), 2)

    def test_square_routine_uses_configured_thirty_centimeter_geometry(self):
        routine = self.library.get(2)
        drive_steps = [step for step in routine.steps if step["motion"] == "drive"]
        self.assertEqual(
            [step["distance_m"] for step in drive_steps],
            [0.3, 0.3, 0.3, 0.3, 0.3],
        )
        self.assertTrue(all(step["speed"] == "linear.fast" for step in drive_steps))

    def test_unbounded_rotate_remains_active(self):
        base = FakeBase()
        executor = SearchingRoutineExecutor(self.config, base, self.library)
        state = {}
        routine = self.library.get(0)
        self.assertFalse(executor.step(routine, state, "search_can"))
        self.assertFalse(executor.step(routine, state, "search_can"))
        self.assertEqual(len(base.starts), 1)
        self.assertEqual(base.active["direction"], "left")

    def test_square_uses_linear_and_turn_calibration_to_advance(self):
        base = FakeBase()
        executor = SearchingRoutineExecutor(self.config, base, self.library)
        executor.library.dry_run_step_seconds = 11.0
        state = {}
        routine = self.library.get(2)

        executor.step(routine, state, "search_can")
        executor.step(routine, state, "search_can")
        self.assertEqual(state[executor.STATE_KEY]["step_index"], 1)
        self.assertIsNone(base.active)

        executor.step(routine, state, "search_can")
        self.assertEqual(base.active["direction"], "left")
        executor.step(routine, state, "search_can")
        self.assertEqual(state[executor.STATE_KEY]["step_index"], 2)

    def test_avoidance_cycle_enters_pose_based_return_to_origin(self):
        vague_map = VagueMap(self.config.section("vague_map"))
        tracker = CommandOdometry(
            vague_map,
            self.config.section("vague_map")["odometry"],
            self.config.section("base_turn_response"),
        )
        base = TrackedFakeBase(tracker)
        executor = SearchingRoutineExecutor(self.config, base, self.library, vague_map=vague_map)
        routine = self.library.get(2)
        state_data = {}
        state = executor._state(routine, state_data)
        state["avoidance_history"].append({"dx_m": 0.6, "dy_m": 0.4, "dheading_rad": 0.0})
        vague_map.robot_pose = Pose2D(0.6, 0.4, 0.0)
        state["step_index"] = len(routine.steps) - 1
        state["progress"] = float(routine.steps[-1]["distance_m"])

        self.assertFalse(executor.step(routine, state_data, "search_can"))
        self.assertEqual(state["phase"], "return_position")
        executor.step(routine, state_data, "search_can")
        self.assertIsNotNone(base.active)
        self.assertIn("_return_", base.active["label"])

        for _ in range(300):
            executor.step(routine, state_data, "search_can")
            if state["phase"] == "route" and state["cycles"] == 1:
                break
        else:
            self.fail("routine recovery did not return to its next cycle")

        tolerance = float(self.config.get("vague_map.navigation.arrival_tolerance_m"))
        self.assertLessEqual(
            math.hypot(vague_map.robot_pose.x_m, vague_map.robot_pose.y_m),
            tolerance,
        )
        heading_error = ((vague_map.robot_pose.heading_rad + math.pi) % (2.0 * math.pi)) - math.pi
        self.assertLessEqual(
            abs(heading_error),
            float(self.config.get("vague_map.navigation.heading_tolerance_rad")),
        )


if __name__ == "__main__":
    unittest.main()
