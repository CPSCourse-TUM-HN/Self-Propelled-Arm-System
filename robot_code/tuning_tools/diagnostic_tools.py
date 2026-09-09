from __future__ import print_function

import csv
import math
import os
import time

import numpy as np


SCSCL_STATUS_FIELDS = (
    "position", "speed", "load", "voltage_raw", "voltage_v",
    "temperature_c", "moving", "current",
)


def find_project_root(start):
    current = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(current, "config.json")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError("could not find project root from {}".format(start))
        current = parent


def decode_signed_magnitude(value, sign_bit):
    value = int(value)
    marker = 1 << int(sign_bit)
    return -(value & ~marker) if value & marker else value


def decode_scscl_status(raw):
    return {
        "position": int(raw["position"]),
        "speed": decode_signed_magnitude(raw["speed"], 15),
        "load": decode_signed_magnitude(raw["load"], 10),
        "voltage_raw": int(raw["voltage"]),
        "voltage_v": float(raw["voltage"]) / 10.0,
        "temperature_c": int(raw["temperature"]),
        "moving": bool(raw["moving"]),
        "current": decode_signed_magnitude(raw["current"], 15),
    }


def _read_register(packet_handler, port_handler, servo_id, address, width):
    reader = packet_handler.read1ByteTxRx if int(width) == 1 else packet_handler.read2ByteTxRx
    value, communication_result, servo_error = reader(port_handler, int(servo_id), int(address))
    if int(communication_result) != 0 or int(servo_error) != 0:
        raise IOError(
            "servo {} register {} read failed communication_result={} servo_error={}".format(
                servo_id, address, communication_result, servo_error
            )
        )
    return int(value)


def read_scscl_status(ttl_servo, servo_id, io_lock=None):
    def read_all():
        packet = ttl_servo.packetHandler
        port = ttl_servo.portHandler
        raw = {
            "position": _read_register(packet, port, servo_id, 56, 2),
            "speed": _read_register(packet, port, servo_id, 58, 2),
            "load": _read_register(packet, port, servo_id, 60, 2),
            "voltage": _read_register(packet, port, servo_id, 62, 1),
            "temperature": _read_register(packet, port, servo_id, 63, 1),
            "moving": _read_register(packet, port, servo_id, 66, 1),
            "current": _read_register(packet, port, servo_id, 69, 2),
        }
        result = decode_scscl_status(raw)
        result["servo_id"] = int(servo_id)
        return result

    if io_lock is None:
        return read_all()
    with io_lock:
        return read_all()


def append_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows:
        return path
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    fieldnames = list(fieldnames or rows[0].keys())
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    return path


def robust_stats(values):
    values = np.asarray([float(value) for value in values], dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": None, "std": None, "median": None, "mad": None}
    median = float(np.median(values))
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": median,
        "mad": float(np.median(np.abs(values - median))),
    }


def pose_from_tag_corners(corners, camera_matrix, dist_coeffs, marker_length_m):
    import cv2

    half = float(marker_length_m) / 2.0
    object_points = np.asarray(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float32,
    )
    image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        np.asarray(camera_matrix, dtype=np.float32),
        np.asarray(dist_coeffs, dtype=np.float32),
        flags=cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed")
    rotation, _ = cv2.Rodrigues(rvec)
    yaw_deg = math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
    pitch_deg = math.degrees(math.atan2(-float(rotation[2, 0]), math.hypot(float(rotation[2, 1]), float(rotation[2, 2]))))
    roll_deg = math.degrees(math.atan2(float(rotation[2, 1]), float(rotation[2, 2])))
    x, y, z = [float(value) for value in tvec.reshape(-1)]
    return {
        "x_m": x,
        "y_m": y,
        "z_m": z,
        "perpendicular_distance_m": abs(z),
        "distance_m": math.sqrt(x * x + y * y + z * z),
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
    }


def summarize_pose_samples(samples, attempted_frames):
    samples = list(samples)
    result = {
        "attempted_frames": int(attempted_frames),
        "detected_frames": len(samples),
        "detection_rate": float(len(samples)) / float(attempted_frames) if attempted_frames else 0.0,
    }
    for field in ("x_m", "y_m", "z_m", "perpendicular_distance_m", "distance_m", "yaw_deg", "pitch_deg", "roll_deg"):
        result[field] = robust_stats([sample[field] for sample in samples if field in sample])
    return result


def candidate_bbox_metrics(analysis, depth_map):
    depth_map = np.asarray(depth_map)
    mask = np.asarray(analysis.mask) if getattr(analysis, "mask", None) is not None else None
    height, width = depth_map.shape[:2]
    rows = []
    for index, bbox in enumerate(getattr(analysis, "candidate_bboxes", None) or []):
        x, y, box_width, box_height = [int(round(value)) for value in bbox]
        left, top = max(0, x), max(0, y)
        right, bottom = min(width, x + box_width), min(height, y + box_height)
        crop = depth_map[top:bottom, left:right]
        if mask is not None:
            valid_mask = mask[top:bottom, left:right] > 0
            values = crop[valid_mask]
        else:
            values = crop.reshape(-1)
        values = values[np.isfinite(values)]
        rows.append({
            "candidate_index": index,
            "bbox": (left, top, max(0, right - left), max(0, bottom - top)),
            "center_x_norm": ((left + right) / 2.0) / float(width),
            "center_y_norm": ((top + bottom) / 2.0) / float(height),
            "area_norm": (max(0, right - left) * max(0, bottom - top)) / float(width * height),
            "median_depth": float(np.median(values)) if values.size else None,
        })
    return rows


class CandidateCountTracker(object):
    def __init__(self, expected_count=2):
        self.expected_count = int(expected_count)
        self.counts = []
        self.previous_centers = None
        self.identity_checks = 0
        self.identity_matches = 0

    def update(self, candidate_count):
        self.counts.append(int(candidate_count))
        return self.summary()

    def update_candidates(self, candidates, maximum_center_shift_norm=0.15):
        candidates = list(candidates)
        self.counts.append(len(candidates))
        centers = sorted(
            [(float(item["center_x_norm"]), float(item["center_y_norm"])) for item in candidates]
        )
        if len(centers) == self.expected_count and self.previous_centers is not None:
            self.identity_checks += 1
            shifts = [
                math.hypot(current[0] - previous[0], current[1] - previous[1])
                for current, previous in zip(centers, self.previous_centers)
            ]
            if all(shift <= float(maximum_center_shift_norm) for shift in shifts):
                self.identity_matches += 1
        self.previous_centers = centers if len(centers) == self.expected_count else None
        return self.summary()

    def summary(self):
        total = len(self.counts)
        expected = sum(1 for count in self.counts if count == self.expected_count)
        merged = sum(1 for count in self.counts if count < self.expected_count)
        split = sum(1 for count in self.counts if count > self.expected_count)
        return {
            "frames": total,
            "expected_count_rate": float(expected) / total if total else 0.0,
            "merge_rate": float(merged) / total if total else 0.0,
            "false_split_rate": float(split) / total if total else 0.0,
            "identity_stability_rate": (
                float(self.identity_matches) / self.identity_checks if self.identity_checks else 0.0
            ),
        }


def calibrate_camera(object_points, image_points, image_size):
    import cv2

    if len(object_points) < 3 or len(object_points) != len(image_points):
        raise ValueError("at least three matched calibration views are required")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, tuple(image_size), None, None
    )
    errors = []
    for object_view, image_view, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(object_view, rvec, tvec, camera_matrix, dist_coeffs)
        errors.append(float(cv2.norm(image_view, projected, cv2.NORM_L2) / len(projected)))
    return {
        "rms": float(rms),
        "mean_reprojection_error_px": float(np.mean(errors)),
        "camera_matrix": camera_matrix,
        "dist_coeff": dist_coeffs,
    }


def timestamped_log_path(project_root, category, filename):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(project_root, "logs", category, stamp, filename)
