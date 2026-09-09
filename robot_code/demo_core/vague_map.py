from __future__ import print_function

import math
import time

from .turn_response import TurnResponseModel


def normalize_heading(angle_rad):
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def linear_response_coefficient(settings, speed):
    """Return calibrated meters per (speed * second), interpolating measured speeds."""
    speed = abs(float(speed))
    samples = []
    for sample in settings.get("linear_response_samples", []):
        sample_speed = abs(float(sample["speed"]))
        coefficient = float(sample["distance_m"]) / (
            sample_speed * float(sample["seconds"])
        )
        samples.append((sample_speed, coefficient))
    samples.sort(key=lambda item: item[0])
    for sample_speed, coefficient in samples:
        if abs(sample_speed - speed) <= 1e-6:
            return coefficient
    for lower, upper in zip(samples, samples[1:]):
        if lower[0] < speed < upper[0]:
            fraction = (speed - lower[0]) / (upper[0] - lower[0])
            return lower[1] + fraction * (upper[1] - lower[1])
    return float(settings["linear_meters_per_speed_second"])


def linear_distance_meters(settings, speed, seconds):
    return (
        abs(float(speed))
        * max(0.0, float(seconds))
        * linear_response_coefficient(settings, speed)
        * float(settings.get("linear_slip_factor", 1.0))
    )


def _rotate_planar(x_m, y_m, heading_rad):
    """Rotate local (+X right, +Y forward) coordinates into map coordinates."""
    cosine = math.cos(float(heading_rad))
    sine = math.sin(float(heading_rad))
    return (
        float(x_m) * cosine + float(y_m) * sine,
        -float(x_m) * sine + float(y_m) * cosine,
    )


def point_in_robot_frame(world_point, robot_pose):
    """Express a world point as (+X right, +Y forward) relative to a robot pose."""
    world_point = Point2D.from_mapping(world_point) if isinstance(world_point, dict) else world_point
    robot_pose = Pose2D.from_mapping(robot_pose) if isinstance(robot_pose, dict) else robot_pose
    return Point2D(*_rotate_planar(
        world_point.x_m - robot_pose.x_m,
        world_point.y_m - robot_pose.y_m,
        -robot_pose.heading_rad,
    ))


def robot_pose_from_tag_pose(marker_pose, observation_pose, camera_mount):
    """Recover a planar base pose from an OpenCV tag pose and a fixed camera mount.

    OpenCV camera +Z is optical forward and +X is image right. Map/base +Y is
    forward and +X is right. Positive heading turns clockwise toward +X.
    marker_pose.heading_rad is the world heading of the
    marker-frame +Z axis used by solvePnP. camera_mount describes the camera
    optical center and +Z heading in the robot base frame.
    """
    marker_pose = Pose2D.from_mapping(marker_pose)
    translation = observation_pose or {}
    rotation = translation.get("rotation_matrix")
    if not isinstance(rotation, (list, tuple)) or len(rotation) != 3:
        raise ValueError("tag pose rotation_matrix must be 3x3")
    if any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in rotation):
        raise ValueError("tag pose rotation_matrix must be 3x3")
    tag_x_camera = float(translation["x"])
    tag_z_camera = float(translation["z"])
    tag_normal_forward = float(rotation[2][2])
    tag_normal_right = float(rotation[0][2])
    if not all(math.isfinite(value) for value in (
        tag_x_camera, tag_z_camera, tag_normal_forward, tag_normal_right
    )):
        raise ValueError("tag pose contains non-finite planar values")
    if math.hypot(tag_normal_forward, tag_normal_right) <= 1e-6:
        raise ValueError("tag normal has no usable ground-plane projection")

    tag_normal_in_camera = math.atan2(tag_normal_right, tag_normal_forward)
    camera_world_heading = normalize_heading(marker_pose.heading_rad - tag_normal_in_camera)
    marker_dx, marker_dy = _rotate_planar(
        tag_x_camera, tag_z_camera, camera_world_heading
    )
    camera_world_x = marker_pose.x_m - marker_dx
    camera_world_y = marker_pose.y_m - marker_dy

    camera_heading_in_base = float(camera_mount.get("heading_rad", 0.0))
    base_heading = normalize_heading(camera_world_heading - camera_heading_in_base)
    camera_offset_x, camera_offset_y = _rotate_planar(
        camera_mount.get("x_m", 0.0), camera_mount.get("y_m", 0.0), base_heading
    )
    return Pose2D(
        camera_world_x - camera_offset_x,
        camera_world_y - camera_offset_y,
        base_heading,
    )


class Point2D(object):
    def __init__(self, x_m=0.0, y_m=0.0):
        self.x_m = float(x_m)
        self.y_m = float(y_m)

    @classmethod
    def from_mapping(cls, value):
        return cls(value.get("x_m", 0.0), value.get("y_m", 0.0))

    def distance_to(self, other):
        return math.hypot(self.x_m - other.x_m, self.y_m - other.y_m)

    def as_dict(self):
        return {"x_m": self.x_m, "y_m": self.y_m}


class Pose2D(Point2D):
    def __init__(self, x_m=0.0, y_m=0.0, heading_rad=0.0):
        Point2D.__init__(self, x_m, y_m)
        self.heading_rad = normalize_heading(heading_rad)

    @classmethod
    def from_mapping(cls, value):
        return cls(
            value.get("x_m", 0.0),
            value.get("y_m", 0.0),
            value.get("heading_rad", 0.0),
        )

    def copy(self):
        return Pose2D(self.x_m, self.y_m, self.heading_rad)

    def as_dict(self):
        result = Point2D.as_dict(self)
        result["heading_rad"] = self.heading_rad
        return result


class MappedCan(object):
    def __init__(self, can_id, position, confidence, timestamp=None):
        self.can_id = int(can_id)
        self.position = position
        self.confidence = float(confidence)
        self.observations = 1
        self.last_seen_at = float(timestamp if timestamp is not None else time.time())

    def merge(self, position, confidence, timestamp=None):
        count = float(self.observations)
        self.position.x_m = (self.position.x_m * count + position.x_m) / (count + 1.0)
        self.position.y_m = (self.position.y_m * count + position.y_m) / (count + 1.0)
        self.observations += 1
        self.confidence = max(self.confidence, float(confidence))
        self.last_seen_at = float(timestamp if timestamp is not None else time.time())

    def as_dict(self):
        return {
            "can_id": self.can_id,
            "position": self.position.as_dict(),
            "confidence": self.confidence,
            "observations": self.observations,
            "last_seen_at": self.last_seen_at,
        }


class VagueMap(object):
    """Runtime-only spatial memory using command-based odometry."""

    def __init__(self, settings):
        self.settings = settings
        self.bounds = dict(settings["bounds_m"])
        self.initial_pose = Pose2D.from_mapping(settings["initial_pose"])
        self.robot_pose = self.initial_pose.copy()
        self.bin_marker_pose = Pose2D.from_mapping(settings["bin_marker_position"])
        self.bin_marker_position = Point2D(self.bin_marker_pose.x_m, self.bin_marker_pose.y_m)
        self.bin_docking_pose = Pose2D.from_mapping(settings["bin_docking_pose"])
        self.bin_side_docking_pose = Pose2D.from_mapping(
            settings.get("bin_side_docking_pose", settings["bin_docking_pose"])
        )
        self.known_cans = {}
        self.selected_can_id = None
        self._next_can_id = 1

    def contains(self, point):
        return (
            float(self.bounds["min_x"]) <= point.x_m <= float(self.bounds["max_x"])
            and float(self.bounds["min_y"]) <= point.y_m <= float(self.bounds["max_y"])
        )

    def set_robot_pose(self, pose, reason="manual"):
        self.robot_pose = pose.copy()
        print("[map] pose reset reason={} pose={}".format(reason, self.robot_pose.as_dict()))

    def correct_robot_pose(self, pose, reason="landmark_correction"):
        """Apply a docking correction and carry remembered can positions with it."""
        corrected_pose = pose.copy()
        previous_pose = self.robot_pose.copy()
        for mapped_can in self.known_cans.values():
            relative = point_in_robot_frame(mapped_can.position, previous_pose)
            offset_x, offset_y = _rotate_planar(
                relative.x_m, relative.y_m, corrected_pose.heading_rad
            )
            mapped_can.position = Point2D(
                corrected_pose.x_m + offset_x,
                corrected_pose.y_m + offset_y,
            )
        self.robot_pose = corrected_pose
        print("[map] pose corrected reason={} pose={} transformed_cans={}".format(
            reason, self.robot_pose.as_dict(), len(self.known_cans)
        ))

    def localize_robot_from_bin_tag(self, observation, camera_mount):
        if not observation or not observation.get("found"):
            return None
        observation_pose = observation.get("pose")
        if not observation_pose:
            return None
        return robot_pose_from_tag_pose(
            self.bin_marker_pose.as_dict(), observation_pose, camera_mount
        )

    def bin_position_in_robot_frame(self):
        return point_in_robot_frame(self.bin_marker_position, self.robot_pose)

    def estimate_can_position(self, observation, depth_value):
        if not observation or not observation.get("found") or depth_value is None:
            return None
        distance_m = float(depth_value) * float(self.settings.get("depth_to_distance_scale_m_per_unit", 1.0))
        if not math.isfinite(distance_m) or distance_m <= 0.0:
            print("[map] can rejected reason=invalid_depth value={}".format(depth_value))
            return None
        bearing_offset = float(observation["error_x"]) * float(self.settings["camera_horizontal_fov_rad"])
        bearing = self.robot_pose.heading_rad + bearing_offset
        return Point2D(
            self.robot_pose.x_m + distance_m * math.sin(bearing),
            self.robot_pose.y_m + distance_m * math.cos(bearing),
        )

    def remember_can(self, position, confidence, timestamp=None):
        if not self.contains(position):
            print("[map] can rejected reason=out_of_bounds position={}".format(position.as_dict()))
            return None
        merge_radius = float(self.settings.get("can_merge_radius_m", 0.3))
        nearest = None
        nearest_distance = None
        for mapped_can in self.known_cans.values():
            distance = mapped_can.position.distance_to(position)
            if nearest_distance is None or distance < nearest_distance:
                nearest = mapped_can
                nearest_distance = distance
        if nearest is not None and nearest_distance <= merge_radius:
            nearest.merge(position, confidence, timestamp)
            print("[map] can merged id={} position={} observations={}".format(
                nearest.can_id, nearest.position.as_dict(), nearest.observations
            ))
            return nearest
        mapped_can = MappedCan(self._next_can_id, position, confidence, timestamp)
        self._next_can_id += 1
        self.known_cans[mapped_can.can_id] = mapped_can
        print("[map] can added id={} position={}".format(mapped_can.can_id, position.as_dict()))
        return mapped_can

    def nearest_can(self):
        if not self.known_cans:
            self.selected_can_id = None
            return None
        mapped_can = min(
            self.known_cans.values(),
            key=lambda item: self.robot_pose.distance_to(item.position),
        )
        self.selected_can_id = mapped_can.can_id
        print("[map] selected can id={} position={}".format(mapped_can.can_id, mapped_can.position.as_dict()))
        return mapped_can

    def remove_can(self, can_id, reason="removed"):
        mapped_can = self.known_cans.pop(int(can_id), None)
        if mapped_can is not None:
            print("[map] can removed id={} reason={}".format(mapped_can.can_id, reason))
        if self.selected_can_id == int(can_id):
            self.selected_can_id = None
        return mapped_can

    def remove_cans_near_robot(self):
        radius = float(self.settings.get("pickup_clear_radius_m", 0.3))
        remove_ids = [
            can_id
            for can_id, mapped_can in self.known_cans.items()
            if self.robot_pose.distance_to(mapped_can.position) <= radius
        ]
        for can_id in remove_ids:
            self.remove_can(can_id, "pickup_clear_radius")
        if self.selected_can_id is not None:
            self.remove_can(self.selected_can_id, "selected_can_picked")

    def snapshot(self):
        return {
            "robot_pose": self.robot_pose.as_dict(),
            "bin_marker_position": self.bin_marker_pose.as_dict(),
            "bin_docking_pose": self.bin_docking_pose.as_dict(),
            "bin_side_docking_pose": self.bin_side_docking_pose.as_dict(),
            "bin_position_in_robot_frame": self.bin_position_in_robot_frame().as_dict(),
            "selected_can_id": self.selected_can_id,
            "known_cans": [item.as_dict() for item in sorted(self.known_cans.values(), key=lambda value: value.can_id)],
        }


class CommandOdometry(object):
    def __init__(self, vague_map, settings, turn_response=None):
        self.vague_map = vague_map
        self.settings = settings
        self.turn_response = TurnResponseModel(
            turn_response,
            settings.get("angular_radians_per_speed_second", math.pi),
        )

    def record_motion(self, direction, speed, effective_seconds):
        direction = str(direction)
        speed = abs(float(speed))
        effective_seconds = max(0.0, float(effective_seconds))
        pose = self.vague_map.robot_pose
        if direction in ("forward", "backward"):
            sign = 1.0 if direction == "forward" else -1.0
            distance = sign * linear_distance_meters(
                self.settings, speed, effective_seconds
            )
            pose.x_m += distance * math.sin(pose.heading_rad)
            pose.y_m += distance * math.cos(pose.heading_rad)
        elif direction in ("left", "right"):
            angle = self.turn_response.angle_radians(direction, speed, effective_seconds)
            pose.heading_rad = normalize_heading(
                pose.heading_rad
                - angle
            )
        else:
            raise ValueError("unknown odometry direction: {}".format(direction))
        print("[map] motion direction={} speed={} effective_seconds={} pose={}".format(
            direction, speed, effective_seconds, pose.as_dict()
        ))
        return pose.copy()


class VagueMapNavigator(object):
    """Coarse point navigation. Visual target handling remains outside this class."""

    def __init__(self, vague_map, base, settings):
        self.vague_map = vague_map
        self.base = base
        self.settings = settings

    def step_toward(self, destination, label, arrival_tolerance_m=None):
        turn_label = "map_turn_{}".format(label)
        turning = hasattr(self.base, "motion_active") and self.base.motion_active(turn_label)
        dry_run_step = self.settings.get(
            "turn_pulse_seconds" if turning else "forward_pulse_seconds",
            0.0,
        )
        self.base.update_motion_odometry(float(dry_run_step))
        pose = self.vague_map.robot_pose
        distance = pose.distance_to(destination)
        tolerance = float(
            arrival_tolerance_m
            if arrival_tolerance_m is not None
            else self.settings.get("arrival_tolerance_m", 0.2)
        )
        if distance <= tolerance:
            self.base.stop()
            print("[map] destination reached label={} distance_m={:.3f}".format(label, distance))
            return True
        desired_heading = math.atan2(destination.x_m - pose.x_m, destination.y_m - pose.y_m)
        heading_error = normalize_heading(desired_heading - pose.heading_rad)
        print("[map] navigate label={} distance_m={:.3f} heading_error_rad={:.3f}".format(
            label, distance, heading_error
        ))
        if abs(heading_error) > float(self.settings["heading_tolerance_rad"]):
            self.base.start_motion(
                "right" if heading_error > 0.0 else "left",
                float(self.settings["turn_speed"]),
                turn_label,
            )
        else:
            self.base.start_motion(
                "forward", float(self.settings["forward_speed"]), "map_forward_{}".format(label)
            )
        return False
