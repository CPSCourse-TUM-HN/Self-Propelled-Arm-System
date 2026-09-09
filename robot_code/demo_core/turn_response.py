from __future__ import print_function

import math


class TurnResponseModel(object):
    """Direction- and speed-specific rotation rates from fixed-angle tests."""

    def __init__(self, settings=None, fallback_radians_per_speed_second=math.pi):
        settings = settings or {}
        self.reference_angle_rad = float(settings.get("reference_angle_rad", math.pi))
        self.fallback_coefficient = float(fallback_radians_per_speed_second)
        self.samples = []
        for sample in settings.get("samples", []):
            self.samples.append({
                "speed": float(sample["speed"]),
                "left_seconds": float(sample["left_seconds"]),
                "right_seconds": float(sample["right_seconds"]),
            })

    def _sample(self, speed):
        speed = abs(float(speed))
        for sample in self.samples:
            if abs(sample["speed"] - speed) <= 1e-6:
                return sample
        return None

    def radians_per_second(self, direction, speed):
        direction = str(direction)
        if direction not in ("left", "right"):
            raise ValueError("turn direction must be left or right")
        speed = abs(float(speed))
        sample = self._sample(speed)
        if sample is not None:
            return self.reference_angle_rad / float(sample[direction + "_seconds"])
        return speed * self.fallback_coefficient

    def angle_radians(self, direction, speed, seconds):
        sign = 1.0 if str(direction) == "left" else -1.0
        return sign * self.radians_per_second(direction, speed) * max(0.0, float(seconds))

    def seconds_for_angle(self, angle_radians, direction, speed):
        rate = self.radians_per_second(direction, speed)
        return abs(float(angle_radians)) / rate

    def matching_seconds(self, source_direction, target_direction, speed, source_seconds):
        angle = abs(self.angle_radians(source_direction, speed, source_seconds))
        return self.seconds_for_angle(angle, target_direction, speed)
