from __future__ import print_function

import os

import cv2
import numpy as np


RAW_FRAME_SPACE = "raw"
RECTIFIED_FRAME_SPACE = "rectified"


class CameraRectifier(object):
    """Calibrated, fixed-size frame rectification with cached OpenCV maps."""

    def __init__(self, config):
        self.config = config
        self.width = int(config.get("camera.width"))
        self.height = int(config.get("camera.height"))
        self.alpha = float(config.get("camera.rectification_alpha", 0.0))
        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError("camera.rectification_alpha must be in [0, 1]")
        path = config.resolve_path(config.get("camera.calibration_yaml"))
        self.calibration_path = path
        self.camera_matrix, self.dist_coeffs = self._load_calibration(path)
        self.rectified_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (self.width, self.height),
            self.alpha,
            (self.width, self.height),
        )
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.rectified_camera_matrix,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

    def _load_calibration(self, path):
        if not path or not os.path.isfile(path):
            raise RuntimeError("camera calibration file is required: {}".format(path))
        try:
            import yaml
        except ImportError:
            raise RuntimeError("PyYAML is required to load camera calibration")
        with open(path, "r") as stream:
            calibration = yaml.safe_load(stream)
        try:
            image_width = int(calibration["image_width"])
            image_height = int(calibration["image_height"])
            matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
            coefficients = np.asarray(calibration["dist_coeff"], dtype=np.float64).reshape(-1)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("invalid camera calibration {}: {}".format(path, exc))
        if (image_width, image_height) != (self.width, self.height):
            raise RuntimeError(
                "camera calibration resolution {}x{} does not match configured {}x{}".format(
                    image_width, image_height, self.width, self.height,
                )
            )
        if matrix.shape != (3, 3) or coefficients.size < 4:
            raise RuntimeError("invalid camera calibration dimensions in {}".format(path))
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(coefficients)):
            raise RuntimeError("camera calibration contains non-finite values: {}".format(path))
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            raise RuntimeError("camera calibration focal lengths must be positive: {}".format(path))
        return matrix, coefficients

    def _validate_frame(self, frame):
        if frame is None:
            return
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            raise RuntimeError(
                "camera frame resolution {}x{} does not match calibration {}x{}".format(
                    frame.shape[1], frame.shape[0], self.width, self.height,
                )
            )

    def rectify(self, frame):
        if frame is None:
            return None
        self._validate_frame(frame)
        return cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR)

    def rectify_points(self, points):
        array = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        corrected = cv2.undistortPoints(
            array,
            self.camera_matrix,
            self.dist_coeffs,
            P=self.rectified_camera_matrix,
        )
        return corrected.reshape(-1, 2)

    def rectify_point(self, x, y):
        point = self.rectify_points([[float(x), float(y)]])[0]
        return float(point[0]), float(point[1])

    def rectify_observation(self, observation):
        """Copy a raw-space bbox observation into this rectified image space."""
        result = dict(observation or {})
        bbox = result.get("bbox")
        if not bbox or len(bbox) != 4:
            result["frame_space"] = RECTIFIED_FRAME_SPACE
            return result
        left, top, right, bottom = [float(value) for value in bbox]
        points = self.rectify_points([
            [left, top], [right, top], [right, bottom], [left, bottom],
        ])
        x_values = points[:, 0]
        y_values = points[:, 1]
        new_left = max(0.0, min(float(self.width), float(np.min(x_values))))
        new_right = max(0.0, min(float(self.width), float(np.max(x_values))))
        new_top = max(0.0, min(float(self.height), float(np.min(y_values))))
        new_bottom = max(0.0, min(float(self.height), float(np.max(y_values))))
        center_x = (new_left + new_right) / 2.0
        center_y = (new_top + new_bottom) / 2.0
        result.update({
            "bbox": [new_left, new_top, new_right, new_bottom],
            "center_x": center_x,
            "center_y": center_y,
            "error_x": (center_x - float(self.width) / 2.0) / float(self.width),
            "bbox_width_px": new_right - new_left,
            "bbox_height_px": new_bottom - new_top,
            "bbox_width_norm": (new_right - new_left) / float(self.width),
            "bbox_height_norm": (new_bottom - new_top) / float(self.height),
            "frame_space": RECTIFIED_FRAME_SPACE,
        })
        return result
