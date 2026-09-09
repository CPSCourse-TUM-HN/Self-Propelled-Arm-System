import math
import unittest

from demo_core import TurnResponseModel, load_config


class TurnResponseModelTest(unittest.TestCase):
    def setUp(self):
        config = load_config()
        self.model = TurnResponseModel(
            config.section("base_turn_response"),
            config.get("vague_map.odometry.angular_radians_per_speed_second"),
        )

    def test_measured_slow_and_fast_rates_are_direction_specific(self):
        self.assertAlmostEqual(self.model.radians_per_second("left", 0.15), (math.pi / 2.0) / 10.4, places=8)
        self.assertAlmostEqual(self.model.radians_per_second("right", 0.15), (math.pi / 2.0) / 9.0, places=8)
        self.assertAlmostEqual(self.model.radians_per_second("left", 0.3), (math.pi / 2.0) / 3.6, places=8)
        self.assertAlmostEqual(self.model.radians_per_second("right", 0.3), (math.pi / 2.0) / 3.4, places=8)

    def test_opposite_turn_time_matches_physical_angle(self):
        right_seconds = self.model.matching_seconds("left", "right", 0.15, 2.5)
        left_angle = abs(self.model.angle_radians("left", 0.15, 2.5))
        right_angle = abs(self.model.angle_radians("right", 0.15, right_seconds))
        self.assertAlmostEqual(right_seconds, 2.5 * 9.0 / 10.4)
        self.assertAlmostEqual(left_angle, right_angle)

    def test_unmeasured_speed_uses_linear_fallback(self):
        self.assertAlmostEqual(self.model.radians_per_second("left", 0.2), 0.2 * math.pi, places=8)


if __name__ == "__main__":
    unittest.main()
