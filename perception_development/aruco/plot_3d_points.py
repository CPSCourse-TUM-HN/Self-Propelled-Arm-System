import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))


def to_camera_center_coordinates(position):
    x, y, z = np.asarray(position, dtype=float)
    return np.array([-x, y, z], dtype=float)


def marker_positions_to_camera_center_coordinates(marker_positions):
    return {
        marker_id: to_camera_center_coordinates(position)
        for marker_id, position in marker_positions.items()
    }


def distance_between_points(point_a, point_b):
    point_a = np.asarray(point_a, dtype=float)
    point_b = np.asarray(point_b, dtype=float)
    return np.linalg.norm(point_b - point_a)


def distances_from_point(points, reference_id):
    if reference_id not in points:
        return {}

    reference_point = points[reference_id]

    return {
        marker_id: distance_between_points(reference_point, point)
        for marker_id, point in points.items()
        if marker_id != reference_id
    }


def relative_vector_between_points(point_a, point_b):
    point_a = np.asarray(point_a, dtype=float)
    point_b = np.asarray(point_b, dtype=float)
    vector = point_b - point_a
    distance = np.linalg.norm(vector)
    horizontal_distance = np.linalg.norm(vector[:2])

    if distance == 0:
        unit_vector = np.zeros(3)
        angle_from_x = 0.0
        angle_from_y = 0.0
        angle_from_z = 0.0
    else:
        unit_vector = vector / distance
        angle_from_x = np.degrees(np.arccos(np.clip(unit_vector[0], -1.0, 1.0)))
        angle_from_y = np.degrees(np.arccos(np.clip(unit_vector[1], -1.0, 1.0)))
        angle_from_z = np.degrees(np.arccos(np.clip(unit_vector[2], -1.0, 1.0)))

    yaw_rad = np.arctan2(vector[1], vector[0])
    pitch_rad = np.arctan2(vector[2], horizontal_distance)

    return {
        "vector": vector,
        "move_claw_vector": vector,
        "distance": distance,
        "unit_vector": unit_vector,
        "move_claw_unit_vector": unit_vector,
        "yaw_degrees": np.degrees(yaw_rad),
        "pitch_degrees": np.degrees(pitch_rad),
        "yaw_sin": np.sin(yaw_rad),
        "yaw_cos": np.cos(yaw_rad),
        "yaw_tan": np.tan(yaw_rad),
        "pitch_sin": np.sin(pitch_rad),
        "pitch_cos": np.cos(pitch_rad),
        "pitch_tan": np.tan(pitch_rad),
        "angle_from_x_degrees": angle_from_x,
        "angle_from_y_degrees": angle_from_y,
        "angle_from_z_degrees": angle_from_z,
        "direction_text": direction_text(vector),
        "move_claw_direction_text": direction_text(vector),
    }


def relative_vectors_from_point(points, reference_id):
    if reference_id not in points:
        return {}

    reference_point = points[reference_id]

    return {
        marker_id: relative_vector_between_points(reference_point, point)
        for marker_id, point in points.items()
        if marker_id != reference_id
    }


def build_claw_guidance(marker_positions, claw_id):
    relative_vectors = relative_vectors_from_point(marker_positions, claw_id)
    guidance = {}

    for marker_id, relative_vector in relative_vectors.items():
        vector = relative_vector["move_claw_vector"]
        unit_vector = relative_vector["move_claw_unit_vector"]

        guidance[int(marker_id)] = {
            "target_id": int(marker_id),
            "claw_id": int(claw_id),
            "distance_m": float(relative_vector["distance"]),
            "move_claw_vector": vector.astype(float).tolist(),
            "move_claw_unit_vector": unit_vector.astype(float).tolist(),
            "yaw_degrees": float(relative_vector["yaw_degrees"]),
            "yaw_sin": float(relative_vector["yaw_sin"]),
            "yaw_cos": float(relative_vector["yaw_cos"]),
            "yaw_tan": float(relative_vector["yaw_tan"]),
            "pitch_degrees": float(relative_vector["pitch_degrees"]),
            "pitch_sin": float(relative_vector["pitch_sin"]),
            "pitch_cos": float(relative_vector["pitch_cos"]),
            "pitch_tan": float(relative_vector["pitch_tan"]),
            "angle_from_x_degrees": float(relative_vector["angle_from_x_degrees"]),
            "angle_from_y_degrees": float(relative_vector["angle_from_y_degrees"]),
            "angle_from_z_degrees": float(relative_vector["angle_from_z_degrees"]),
            "move_claw_direction_text": relative_vector["move_claw_direction_text"],
        }

    return guidance


def format_marker_pose(marker_id, position):
    x, y, z = np.asarray(position, dtype=float)
    distance = distance_between_points((0, 0, 0), position)
    return f"ID = {marker_id}, X = {x:.3f} m, Y = {y:.3f} m, Z(depth) = {z:.3f} m, Distance = {distance:.3f} m"


def format_claw_guidance(claw_guidance):
    lines = []

    for marker_id, guidance in claw_guidance.items():
        vector = guidance["move_claw_vector"]
        unit_vector = guidance["move_claw_unit_vector"]
        lines.append(
            f"Claw ID {guidance['claw_id']} -> ID {marker_id}: "
            f"distance = {guidance['distance_m']:.3f} m, "
            f"move claw vector = [{vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}], "
            f"move claw unit = [{unit_vector[0]:.3f}, {unit_vector[1]:.3f}, {unit_vector[2]:.3f}], "
            f"yaw = {guidance['yaw_degrees']:.1f} deg "
            f"(sin={guidance['yaw_sin']:.3f}, cos={guidance['yaw_cos']:.3f}, tan={guidance['yaw_tan']:.3f}), "
            f"pitch = {guidance['pitch_degrees']:.1f} deg "
            f"(sin={guidance['pitch_sin']:.3f}, cos={guidance['pitch_cos']:.3f}, tan={guidance['pitch_tan']:.3f}), "
            f"move claw: {guidance['move_claw_direction_text']}"
        )

    return lines


def direction_text(vector):
    dx, dy, dz = np.asarray(vector, dtype=float)
    x_direction = "left" if dx > 0 else "right" if dx < 0 else "same-x"
    y_direction = "down" if dy > 0 else "up" if dy < 0 else "same-y"
    z_direction = "forward/deeper" if dz > 0 else "back/closer" if dz < 0 else "same-z"
    return f"{x_direction}, {y_direction}, {z_direction}"


class Aruco3DPlotter:
    def __init__(
        self,
        title="ArUco 3D Points",
        pause_seconds=0.001,
        camera_bounds=((-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)),
    ):
        self.title = title
        self.pause_seconds = pause_seconds
        self.camera_bounds = camera_bounds
        self.figure = None
        self.axes = None
        self.plt = None

    def update(self, marker_positions, claw_id=None):
        points = {
            int(marker_id): np.asarray(position, dtype=float)
            for marker_id, position in marker_positions.items()
        }

        if not points:
            return

        self._ensure_plot()
        self.axes.clear()

        relative_vectors = relative_vectors_from_point(points, claw_id) if claw_id is not None else {}

        for marker_id, point in points.items():
            color = "red" if marker_id == claw_id else "blue"
            self.axes.scatter(point[0], point[1], point[2], color=color, s=60)
            self.axes.text(
                point[0],
                point[1],
                point[2],
                f"ID {marker_id}\nx={point[0]:.3f}\ny={point[1]:.3f}\nz={point[2]:.3f}",
            )

        if claw_id in points:
            claw_point = points[claw_id]

            for marker_id, relative_vector in relative_vectors.items():
                marker_point = points[marker_id]
                midpoint = (claw_point + marker_point) / 2
                vector = relative_vector["vector"]
                distance = relative_vector["distance"]

                self.axes.plot(
                    [claw_point[0], marker_point[0]],
                    [claw_point[1], marker_point[1]],
                    [claw_point[2], marker_point[2]],
                    color="green",
                )
                self.axes.quiver(
                    claw_point[0],
                    claw_point[1],
                    claw_point[2],
                    vector[0],
                    vector[1],
                    vector[2],
                    color="orange",
                    arrow_length_ratio=0.15,
                )
                self.axes.text(
                    midpoint[0],
                    midpoint[1],
                    midpoint[2],
                    f"{distance:.3f} m\nyaw={relative_vector['yaw_degrees']:.1f}°\npitch={relative_vector['pitch_degrees']:.1f}°",
                    color="green",
                )

        self.axes.set_title(self.title)
        self.axes.set_xlabel("X (m, left + / right -)")
        self.axes.set_ylabel("Y (m, down + / up -)")
        self.axes.set_zlabel("Z (m, forward +)")
        self._set_axes(points.values())

        self.figure.canvas.draw_idle()
        self.plt.pause(self.pause_seconds)

    def close(self):
        if self.figure is not None:
            self.plt.close(self.figure)
            self.figure = None
            self.axes = None

    def _ensure_plot(self):
        if self.plt is None:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D as _Axes3D

            self.plt = plt

        if self.figure is not None and self.plt.fignum_exists(self.figure.number):
            return

        self.plt.ion()
        self.figure = self.plt.figure(self.title)
        self.axes = self.figure.add_subplot(111, projection="3d")

    def _set_axes(self, points):
        if self.camera_bounds is not None:
            x_bounds, y_bounds, z_bounds = self.camera_bounds
            self.axes.set_xlim(x_bounds)
            self.axes.set_ylim(y_bounds)
            self.axes.set_zlim(z_bounds)
            return

        points = np.array(list(points), dtype=float)
        center = points.mean(axis=0)
        point_range = np.ptp(points, axis=0)
        radius = max(np.max(point_range) / 2, 0.05)

        self.axes.set_xlim(center[0] - radius, center[0] + radius)
        self.axes.set_ylim(center[1] - radius, center[1] + radius)
        self.axes.set_zlim(center[2] - radius, center[2] + radius)
