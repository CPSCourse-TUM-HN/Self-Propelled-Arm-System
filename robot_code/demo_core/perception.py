import math
import os
import sys
import threading
import time

import cv2
import numpy as np

from .camera_geometry import CameraRectifier, RAW_FRAME_SPACE, RECTIFIED_FRAME_SPACE

from .depth_vision import CameraOnly, DepthCamera, setup_jetson_inference_paths, summarize_region


def empty_detection(kind):
    return {
        "kind": kind,
        "found": False,
        "confidence": 0.0,
        "bbox": None,
        "center_x": None,
        "center_y": None,
        "error_x": None,
        "timestamp": time.time(),
    }


def bbox_center_result(kind, bbox, image_width, image_height, confidence=1.0, extra=None):
    left, top, right, bottom = bbox
    center_x = (float(left) + float(right)) / 2.0
    center_y = (float(top) + float(bottom)) / 2.0
    bbox_width = max(0.0, float(right) - float(left))
    bbox_height = max(0.0, float(bottom) - float(top))
    result = {
        "kind": kind,
        "found": True,
        "confidence": float(confidence),
        "bbox": [float(left), float(top), float(right), float(bottom)],
        "center_x": center_x,
        "center_y": center_y,
        "error_x": (center_x - float(image_width) / 2.0) / float(image_width),
        "bbox_width_px": bbox_width,
        "bbox_height_px": bbox_height,
        "bbox_width_norm": bbox_width / float(max(1, int(image_width))),
        "bbox_height_norm": bbox_height / float(max(1, int(image_height))),
        "timestamp": time.time(),
    }
    if extra:
        result.update(extra)
    return result


def tag_size_metrics(points, image_width, image_height):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    edge_lengths = []
    edge_midpoints = []
    edge_vectors = []
    for index in range(4):
        delta = pts[(index + 1) % 4] - pts[index]
        edge_lengths.append(float(np.linalg.norm(delta)))
        edge_midpoints.append((pts[(index + 1) % 4] + pts[index]) / 2.0)
        edge_vectors.append(delta)
    edge_mean = float(np.mean(edge_lengths))
    normalizer = float(max(1, min(int(image_width), int(image_height))))
    vertical = sorted(
        [index for index, delta in enumerate(edge_vectors) if abs(float(delta[1])) >= abs(float(delta[0]))],
        key=lambda index: float(edge_midpoints[index][0]),
    )
    horizontal = sorted(
        [index for index, delta in enumerate(edge_vectors) if abs(float(delta[0])) > abs(float(delta[1]))],
        key=lambda index: float(edge_midpoints[index][1]),
    )
    vertical_error = 0.0
    horizontal_error = 0.0
    if len(vertical) == 2:
        left_length = edge_lengths[vertical[0]]
        right_length = edge_lengths[vertical[1]]
        vertical_error = (right_length - left_length) / max(1.0, (right_length + left_length) / 2.0)
    if len(horizontal) == 2:
        top_length = edge_lengths[horizontal[0]]
        bottom_length = edge_lengths[horizontal[1]]
        horizontal_error = (bottom_length - top_length) / max(1.0, (bottom_length + top_length) / 2.0)
    angle_errors = []
    for index in range(4):
        before = pts[(index - 1) % 4] - pts[index]
        after = pts[(index + 1) % 4] - pts[index]
        denominator = float(np.linalg.norm(before) * np.linalg.norm(after))
        if denominator <= 1e-6:
            angle_errors.append(90.0)
            continue
        cosine = max(-1.0, min(1.0, float(np.dot(before, after)) / denominator))
        angle_errors.append(abs(math.degrees(math.acos(cosine)) - 90.0))
    return {
        "corners": [[float(x), float(y)] for x, y in pts],
        "edge_lengths_px": edge_lengths,
        "edge_length_px": edge_mean,
        "edge_length_norm": edge_mean / normalizer,
        "vertical_edge_error": float(vertical_error),
        "horizontal_edge_error": float(horizontal_error),
        "max_corner_angle_error_deg": float(max(angle_errors)),
    }


def tag_yaw_from_rotation_matrix(rotation_matrix):
    """Return signed left/right Tag-plane yaw around the camera Y axis."""
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        return None
    normal = rotation[:, 2].copy()
    if normal[2] < 0.0:
        normal *= -1.0
    if abs(float(normal[0])) + abs(float(normal[2])) <= 1e-9:
        return None
    return float(math.atan2(float(normal[0]), float(normal[2])))


class CanDetector(object):
    def __init__(self, config):
        self.config = config
        self.settings = config.section("detectors")["can"]
        self.net = None
        self.dry_run_target_available = True
        self.backend = "detectnet_native"
        self._cuda_from_numpy = None
        self.rectification_enabled = bool(self.settings.get("rectification_enabled", False))
        self.rectifier = CameraRectifier(config) if self.rectification_enabled else None

    @property
    def frame_space(self):
        return RECTIFIED_FRAME_SPACE if self.rectification_enabled else RAW_FRAME_SPACE

    def prepare_frame(self, frame):
        if frame is None or self.rectifier is None:
            return frame
        return self.rectifier.rectify(frame)

    def display_frame(self, frame):
        """Return the image space in which this detector reports its bboxes."""
        return self.prepare_frame(frame)

    def observation_in_display_space(self, observation):
        if self.rectifier is None or not observation:
            return observation
        if observation.get("frame_space", RAW_FRAME_SPACE) == RECTIFIED_FRAME_SPACE:
            return observation
        return self.rectifier.rectify_observation(observation)

    def confidence_threshold(self, tracking=False):
        key = "tracking_confidence_threshold" if tracking else "confidence_threshold"
        return float(self.settings.get(key, self.settings["confidence_threshold"]))

    def reset(self):
        self.net = None
        self._cuda_from_numpy = None
        self.dry_run_target_available = True

    def load(self):
        if not self.settings.get("enabled", True):
            print("[can] enabled=False; using placeholder detector")
            return
        if self.config.get("runtime.dry_run.camera", True):
            print("[can] camera dry-run; skip can model load")
            return
        if self.net is not None:
            return
        print("[can] rectification enabled={} frame_space={}".format(
            self.rectification_enabled, self.frame_space,
        ))
        setup_jetson_inference_paths()

        model_path = self.config.resolve_path(self.settings["model_path"])
        labels_path = self.config.resolve_path(self.settings["labels_path"])
        if not os.path.exists(model_path):
            raise RuntimeError("can model not found: {}".format(model_path))
        if not os.path.exists(labels_path):
            raise RuntimeError("can labels not found: {}".format(labels_path))

        self._load_detectnet_native(model_path, labels_path)

    def _load_detectnet_native(self, model_path, labels_path):
        from jetson_inference import detectNet
        from jetson_utils import cudaFromNumpy

        print("[can] loading native detectNet model={}".format(model_path))
        self.net = detectNet(
            model=model_path,
            labels=labels_path,
            input_blob=self.settings.get("input_blob", "input_0"),
            output_cvg=self.settings.get("output_scores", "scores"),
            output_bbox=self.settings.get("output_boxes", "boxes"),
            threshold=self.confidence_threshold(tracking=True),
        )
        clustering = float(self.settings.get("clustering_threshold", 0.3))
        if hasattr(self.net, "SetClusteringThreshold"):
            self.net.SetClusteringThreshold(clustering)
        else:
            raise RuntimeError("detectNet binding does not expose SetClusteringThreshold")
        self._cuda_from_numpy = cudaFromNumpy
        print("[can] native detectNet loaded threshold={:.3f} clustering={:.3f}".format(
            self.confidence_threshold(tracking=True),
            clustering,
        ))

    def detect(self, frame):
        detections = self.detect_all(frame)
        if not detections:
            return empty_detection("can")
        best = max(detections, key=lambda detection: float(detection["confidence"]))
        print(
            "[can] backend={} confidence={:.3f} class_id={} center=({:.1f},{:.1f}) error_x={:.3f} bbox={}".format(
                best.get("backend", self.backend),
                best["confidence"],
                best.get("class_id", int(self.settings.get("class_id", 1))),
                best["center_x"],
                best["center_y"],
                best["error_x"],
                best["bbox"],
            )
        )
        return best

    def detect_all(self, frame):
        if frame is None:
            print("[can] no frame")
            return []
        frame = self.prepare_frame(frame)
        if self.config.get("runtime.dry_run.camera", True):
            if not self.dry_run_target_available:
                return []
            height, width = frame.shape[:2]
            return [bbox_center_result(
                "can",
                [width * 0.35, height * 0.15, width * 0.65, height * 0.85],
                width,
                height,
                confidence=0.95,
                extra={"simulated": True, "frame_space": self.frame_space},
            )]
        if self.net is None:
            if self.settings.get("enabled", True):
                self.load()
            else:
                print("[can] placeholder: no target detection available")
                return []

        return self._detect_all_native(frame)

    def mark_dry_run_target_consumed(self):
        if self.config.get("runtime.dry_run.camera", True):
            self.dry_run_target_available = False

    def _detect_all_native(self, frame):
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = self.net.Detect(self._cuda_from_numpy(rgb))
        expected_class = int(self.settings.get("class_id", 1))
        detections = [
            detection
            for detection in detections
            if int(detection.ClassID) == expected_class
        ]
        if not detections:
            print("[can] backend=detectnet_native no detections")
            return []

        image_height, image_width = frame.shape[:2]
        results = []
        for detection in detections:
            results.append(bbox_center_result(
                "can",
                [detection.Left, detection.Top, detection.Right, detection.Bottom],
                image_width,
                image_height,
                confidence=detection.Confidence,
                extra={
                    "class_id": int(detection.ClassID),
                    "count": len(detections),
                    "raw_outputs": False,
                    "backend": self.backend,
                    "frame_space": self.frame_space,
                },
            ))
        print(
            "[can] backend={} detections={} best_confidence={:.3f}".format(
                self.backend,
                len(results),
                max(float(result["confidence"]) for result in results),
            )
        )
        return results


class AprilTagBinDetector(object):
    def __init__(self, config):
        self.config = config
        self.settings = config.section("detectors")["bin"]
        self.detector = None
        self.aruco_dict = None
        self.backend = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.parameters = None

    def check_dependency(self):
        has_pupil = False
        pupil_error = None
        try:
            from pupil_apriltags import Detector  # noqa: F401
            has_pupil = True
        except Exception as exc:
            pupil_error = exc
        has_aruco = hasattr(cv2, "aruco")
        has_detector = has_aruco and hasattr(cv2.aruco, "ArucoDetector")
        dictionary = self.settings.get("dictionary", "DICT_APRILTAG_36h11")
        has_tag = has_aruco and hasattr(cv2.aruco, dictionary)
        print("[tag] cv2={}".format(cv2.__version__))
        print("[tag] has pupil_apriltags={}".format(has_pupil))
        if pupil_error is not None:
            print("[tag] pupil_apriltags error={}".format(pupil_error))
        print("[tag] has aruco={}".format(has_aruco))
        print("[tag] has ArucoDetector={}".format(has_detector))
        print("[tag] has {}={}".format(dictionary, has_tag))
        return has_pupil or (has_aruco and has_tag)

    def load(self):
        if self.detector is not None or self.aruco_dict is not None:
            return
        if self.config.get("runtime.dry_run.camera", True):
            print("[tag] camera dry-run; skip detector load")
            return
        if not self.check_dependency():
            raise RuntimeError("AprilTag support is missing; install pupil-apriltags or OpenCV aruco")

        requested_backend = self.settings.get("backend", "pupil_apriltags")
        if requested_backend in ("pupil_apriltags", "auto"):
            try:
                from pupil_apriltags import Detector
                self.detector = Detector(
                    families=self.settings.get("family", "tag36h11"),
                    nthreads=2,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0,
                )
                self.backend = "pupil_apriltags"
            except Exception as exc:
                if requested_backend == "pupil_apriltags":
                    raise RuntimeError("pupil_apriltags load failed: {}".format(exc))
                print("[tag] pupil_apriltags unavailable; trying cv2.aruco: {}".format(exc))

        if self.backend is None:
            dictionary = self.settings.get("dictionary", "DICT_APRILTAG_36h11")
            if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, dictionary):
                raise RuntimeError("OpenCV aruco/AprilTag support is missing")
            dict_id = getattr(cv2.aruco, dictionary)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            if hasattr(cv2.aruco, "DetectorParameters"):
                parameters = cv2.aruco.DetectorParameters()
            else:
                parameters = cv2.aruco.DetectorParameters_create()
            if hasattr(cv2.aruco, "ArucoDetector"):
                self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, parameters)
            else:
                self.detector = None
                self.parameters = parameters
            self.backend = "cv2.aruco"
        self._load_calibration()
        print("[tag] detector loaded backend={} family={} dictionary={}".format(
            self.backend,
            self.settings.get("family", "tag36h11"),
            self.settings.get("dictionary", "DICT_APRILTAG_36h11"),
        ))

    def _load_calibration(self):
        path = self.config.resolve_path(self.config.get("camera.calibration_yaml"))
        if not path or not os.path.exists(path):
            raise RuntimeError("AprilTag calibration file is required: {}".format(path))
        try:
            import yaml
        except ImportError:
            raise RuntimeError("PyYAML is required to load AprilTag camera calibration")
        with open(path, "r") as f:
            calib = yaml.safe_load(f)
        try:
            self.camera_matrix = np.array(calib["camera_matrix"], dtype=np.float32)
            self.dist_coeffs = np.array(calib["dist_coeff"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("invalid AprilTag calibration file {}: {}".format(path, exc))
        if self.camera_matrix.shape != (3, 3) or self.dist_coeffs.size < 4:
            raise RuntimeError("invalid AprilTag calibration dimensions in {}".format(path))
        print("[tag] calibration loaded {}".format(path))

    def detect(self, frame):
        if frame is None:
            print("[tag] no frame")
            return empty_detection("bin_tag")
        if self.config.get("runtime.dry_run.camera", True):
            height, width = frame.shape[:2]
            points = np.array([
                [width * 0.35, height * 0.25],
                [width * 0.65, height * 0.25],
                [width * 0.65, height * 0.75],
                [width * 0.35, height * 0.75],
            ], dtype=np.float32)
            extra = {
                "id": int(self.settings["tag_id"]),
                "simulated": True,
                "frame_space": RAW_FRAME_SPACE,
                "pose": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.19,
                    "rvec": [0.0, 0.0, 0.0],
                    "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                },
                "distance": 0.19,
                "yaw_error_rad": 0.0,
                "yaw_error_deg": 0.0,
                "yaw_source": "simulated",
            }
            extra.update(tag_size_metrics(points, width, height))
            return bbox_center_result(
                "bin_tag",
                [width * 0.35, height * 0.25, width * 0.65, height * 0.75],
                width,
                height,
                confidence=1.0,
                extra=extra,
            )
        self.load()
        if self.backend == "pupil_apriltags":
            return self._detect_with_pupil(frame)
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(frame)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.parameters)
        if ids is None or len(ids) == 0:
            print("[tag] no detections")
            return empty_detection("bin_tag")

        target_id = int(self.settings["tag_id"])
        image_height, image_width = frame.shape[:2]
        best_result = None
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            if marker_id != target_id:
                continue
            pts = marker_corners[0].astype(np.float32)
            left = float(np.min(pts[:, 0]))
            right = float(np.max(pts[:, 0]))
            top = float(np.min(pts[:, 1]))
            bottom = float(np.max(pts[:, 1]))
            extra = {
                "id": marker_id,
                "pose": None,
                "distance": None,
                "frame_space": RAW_FRAME_SPACE,
            }
            extra.update(tag_size_metrics(pts, image_width, image_height))
            pose = self._estimate_pose(pts)
            if pose is not None:
                extra.update(pose)
            extra.update(self._estimate_yaw(pts, image_width, image_height, pose))
            best_result = bbox_center_result(
                "bin_tag",
                [left, top, right, bottom],
                image_width,
                image_height,
                confidence=1.0,
                extra=extra,
            )
            break

        if best_result is None:
            print("[tag] target id={} not found ids={}".format(target_id, [int(x) for x in ids.flatten()]))
            return empty_detection("bin_tag")

        print(
            "[tag] id={} center=({:.1f},{:.1f}) error_x={:.3f} yaw_deg={} height={:.1f}px height_norm={:.3f} distance={}".format(
                best_result["id"],
                best_result["center_x"],
                best_result["center_y"],
                best_result["error_x"],
                None if best_result.get("yaw_error_deg") is None else round(best_result["yaw_error_deg"], 2),
                best_result["bbox_height_px"],
                best_result["bbox_height_norm"],
                best_result["distance"],
            )
        )
        return best_result

    def _detect_with_pupil(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray, estimate_tag_pose=False)
        if not detections:
            print("[tag] no detections")
            return empty_detection("bin_tag")

        target_id = int(self.settings["tag_id"])
        image_height, image_width = frame.shape[:2]
        best_result = None
        seen_ids = []
        for tag in detections:
            marker_id = int(tag.tag_id)
            seen_ids.append(marker_id)
            if marker_id != target_id:
                continue
            pts = tag.corners.astype(np.float32)
            left = float(np.min(pts[:, 0]))
            right = float(np.max(pts[:, 0]))
            top = float(np.min(pts[:, 1]))
            bottom = float(np.max(pts[:, 1]))
            extra = {
                "id": marker_id,
                "pose": None,
                "distance": None,
                "decision_margin": float(getattr(tag, "decision_margin", 0.0)),
                "hamming": int(getattr(tag, "hamming", 0)),
                "backend": self.backend,
                "frame_space": RAW_FRAME_SPACE,
            }
            extra.update(tag_size_metrics(pts, image_width, image_height))
            pose = self._estimate_pose(pts)
            if pose is not None:
                extra.update(pose)
            extra.update(self._estimate_yaw(pts, image_width, image_height, pose))
            best_result = bbox_center_result(
                "bin_tag",
                [left, top, right, bottom],
                image_width,
                image_height,
                confidence=1.0,
                extra=extra,
            )
            break

        if best_result is None:
            print("[tag] target id={} not found ids={}".format(target_id, seen_ids))
            return empty_detection("bin_tag")

        print(
            "[tag] backend={} id={} center=({:.1f},{:.1f}) error_x={:.3f} yaw_deg={} height={:.1f}px height_norm={:.3f} margin={:.2f} distance={}".format(
                best_result["backend"],
                best_result["id"],
                best_result["center_x"],
                best_result["center_y"],
                best_result["error_x"],
                None if best_result.get("yaw_error_deg") is None else round(best_result["yaw_error_deg"], 2),
                best_result["bbox_height_px"],
                best_result["bbox_height_norm"],
                best_result["decision_margin"],
                best_result["distance"],
            )
        )
        return best_result

    def _estimate_yaw(self, image_points, image_width, image_height, pose_result=None):
        rotation = None
        if pose_result is not None:
            rotation = (pose_result.get("pose") or {}).get("rotation_matrix")
        yaw = tag_yaw_from_rotation_matrix(rotation) if rotation is not None else None
        source = "calibrated_pnp"
        return {
            "yaw_error_rad": yaw,
            "yaw_error_deg": None if yaw is None else math.degrees(yaw),
            "yaw_source": source if yaw is not None else None,
        }

    def _estimate_pose(self, image_points):
        if self.camera_matrix is None or self.dist_coeffs is None:
            return None
        marker_length = float(self.settings.get("marker_length_m", 0.08))
        obj_points = np.array(
            [
                [-marker_length / 2.0, marker_length / 2.0, 0],
                [marker_length / 2.0, marker_length / 2.0, 0],
                [marker_length / 2.0, -marker_length / 2.0, 0],
                [-marker_length / 2.0, -marker_length / 2.0, 0],
            ],
            dtype=np.float32,
        )
        ok, rvec, tvec = cv2.solvePnP(obj_points, image_points, self.camera_matrix, self.dist_coeffs)
        if not ok:
            return None
        x, y, z = [float(v) for v in tvec.flatten()]
        distance = float(math.sqrt(x * x + y * y + z * z))
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        return {
            "pose": {
                "x": x,
                "y": y,
                "z": z,
                "rvec": [float(value) for value in rvec.flatten()],
                "rotation_matrix": [
                    [float(value) for value in row]
                    for row in rotation_matrix
                ],
            },
            "distance": distance,
        }


class DepthSensor(object):
    def __init__(self, config):
        self.config = config
        self.camera_settings = config.section("camera")
        self.avoidance_settings = config.section("avoidance")
        self.depth = None
        self._inference_lock = threading.RLock()
        self._frame_lock = threading.Lock()
        self.latest_lens_stats = None
        self.depth_rectification_enabled = bool(
            self.camera_settings.get("depth_rectification_enabled", False)
        )
        self.rectifier = CameraRectifier(config) if self.depth_rectification_enabled else None

    @property
    def depth_frame_space(self):
        return RECTIFIED_FRAME_SPACE if self.rectifier is not None else RAW_FRAME_SPACE

    def depth_input_frame(self, frame):
        """Return the RGB frame whose pixels align with the produced depth map."""
        if frame is None or self.rectifier is None:
            return frame
        return self.rectifier.rectify(frame)

    def point_in_depth_space(self, center_x, center_y, source_space=RAW_FRAME_SPACE):
        if center_x is None or center_y is None:
            return center_x, center_y
        if self.rectifier is not None and source_space != RECTIFIED_FRAME_SPACE:
            return self.rectifier.rectify_point(center_x, center_y)
        return float(center_x), float(center_y)

    def error_x_in_depth_space(self, observation, image_width, image_height):
        if not observation:
            return 0.0
        center_x, _ = self.point_in_depth_space(
            observation.get("center_x"),
            observation.get("center_y", float(image_height) / 2.0),
            observation.get("frame_space", RAW_FRAME_SPACE),
        )
        if center_x is None:
            return float(observation.get("error_x", 0.0))
        return (float(center_x) - float(image_width) / 2.0) / float(max(1, image_width))

    def start(self, camera_only=False):
        if self.config.get("runtime.dry_run.camera", True):
            print("[depth] camera dry-run")
            return

        depth_requested = bool(self.camera_settings.get("depth_enabled", True)) and not camera_only
        if self.depth is not None:
            if camera_only:
                print("[camera] already started depth_enabled={}".format(self.is_depth_available()))
                return
            if self.is_depth_available() == depth_requested:
                print("[camera] already started depth_enabled={}".format(self.is_depth_available()))
                return
            print("[camera] switching depth_enabled={} -> {}".format(
                self.is_depth_available(), depth_requested
            ))
            self.stop()

        if depth_requested:
            self.depth = DepthCamera(
                width=int(self.camera_settings["width"]),
                height=int(self.camera_settings["height"]),
                network=str(self.camera_settings.get("depth_network", "fcn-mobilenet")),
                rectifier=self.rectifier,
            )
        else:
            self.depth = CameraOnly(
                width=int(self.camera_settings["width"]),
                height=int(self.camera_settings["height"]),
            )
        self.depth.start(warmup_frames=2)
        print("[depth] started enabled={} rectification={} frame_space={}".format(
            self.is_depth_available(), self.depth_rectification_enabled, self.depth_frame_space,
        ))

    def is_depth_available(self):
        return bool(self.depth is not None and getattr(self.depth, "depth_enabled", False))

    def read_frame(self):
        with self._frame_lock:
            if self.depth is None:
                if self.config.get("runtime.dry_run.camera", True):
                    return np.zeros(
                        (int(self.camera_settings["height"]), int(self.camera_settings["width"]), 3),
                        dtype=np.uint8,
                    )
                return None
            frame = self.depth.read_frame()
            return None if frame is None else np.array(frame, copy=True)

    def stop(self):
        with self._inference_lock:
            with self._frame_lock:
                if self.depth is not None:
                    try:
                        self.depth.stop()
                    finally:
                        self.depth = None
        print("[depth] stop")

    def _roi(self, roi_name):
        if roi_name == "obstacle_depth_roi":
            return self.avoidance_settings.get("roi", [0.35, 0.52, 0.65, 0.92])
        raise KeyError("unknown depth ROI: {}".format(roi_name))

    def observe_frame(self, roi_name, frame):
        roi = self._roi(roi_name)
        if self.depth is not None and not self.is_depth_available():
            print("[depth] {} unavailable; DepthNet disabled".format(roi_name))
            return None
        if self.depth is None:
            value = 2.0
            print("[depth] placeholder {} mean={:.3f}".format(roi_name, value))
            return {"mean": value, "min": value, "max": value, "roi": roi}
        with self._inference_lock:
            stats = self.depth.observe(region=tuple(roi), frame=frame)
        return self._report_stats(roi_name, stats)

    def observe_center_frame(
        self, label, frame, center_x, center_y, width_ratio, height_ratio,
        report=True, source_space=RAW_FRAME_SPACE,
    ):
        if self.depth is not None and not self.is_depth_available():
            print("[depth] {} unavailable; DepthNet disabled".format(label))
            return None
        if frame is None or center_x is None or center_y is None:
            print("[depth] {} missing frame/center".format(label))
            return None
        image_height, image_width = frame.shape[:2]
        if image_width <= 0 or image_height <= 0:
            print("[depth] {} invalid frame shape".format(label))
            return None

        center_x, center_y = self.point_in_depth_space(center_x, center_y, source_space)
        cx = max(0.0, min(1.0, float(center_x) / float(image_width)))
        cy = max(0.0, min(1.0, float(center_y) / float(image_height)))
        half_w = max(0.005, float(width_ratio) / 2.0)
        half_h = max(0.005, float(height_ratio) / 2.0)
        roi = (
            max(0.0, cx - half_w),
            max(0.0, cy - half_h),
            min(1.0, cx + half_w),
            min(1.0, cy + half_h),
        )
        if self.depth is None:
            value = 2.0
            print("[depth] placeholder {} mean={:.3f} roi={}".format(label, value, roi))
            return {"mean": value, "min": value, "max": value, "roi": roi}
        with self._inference_lock:
            stats = self.depth.observe(region=roi, frame=frame)
        if stats is not None:
            stats["roi"] = roi
            stats["source_center"] = (float(center_x), float(center_y))
        return self._report_stats(label, stats) if report else stats

    def observe_lens_center_frame(self, frame, report=True):
        if frame is None:
            print("[depth] lens_center_depth_roi missing frame")
            return None
        image_height, image_width = frame.shape[:2]
        stats = self.observe_center_frame(
            "lens_center_depth_roi",
            frame,
            float(image_width) / 2.0,
            float(image_height) / 2.0,
            float(self.camera_settings.get("lens_roi_width", 0.12)),
            float(self.camera_settings.get("lens_roi_height", 0.18)),
            report=report,
        )
        if stats is not None:
            self.latest_lens_stats = dict(stats)
        return stats

    def depth_map_frame(self, frame, frame_space=RAW_FRAME_SPACE):
        """Return the full current depth field for local path planning."""
        if frame is None:
            return None
        if self.depth is None and self.config.get("runtime.dry_run.camera", True):
            return np.full(frame.shape[:2], 2.0, dtype=np.float32)
        if not self.is_depth_available() or self.depth is None:
            return None
        with self._inference_lock:
            return self.depth.process_frame(frame, frame_space=frame_space)

    def observe_center_depth_map(
        self,
        label,
        depth_map,
        center_x,
        center_y,
        image_width,
        image_height,
        width_ratio,
        height_ratio,
        source_space=RAW_FRAME_SPACE,
    ):
        if depth_map is None or center_x is None or center_y is None:
            return None
        center_x, center_y = self.point_in_depth_space(center_x, center_y, source_space)
        cx = max(0.0, min(1.0, float(center_x) / float(max(1, int(image_width)))))
        cy = max(0.0, min(1.0, float(center_y) / float(max(1, int(image_height)))))
        half_w = max(0.005, float(width_ratio) / 2.0)
        half_h = max(0.005, float(height_ratio) / 2.0)
        stats = summarize_region(
            depth_map,
            max(0.0, cx - half_w),
            max(0.0, cy - half_h),
            min(1.0, cx + half_w),
            min(1.0, cy + half_h),
        )
        return self._report_stats(label, stats)

    def _report_stats(self, roi_name, stats):
        if stats is None:
            print("[depth] {} no frame".format(roi_name))
            return None
        print(
            "[depth] {} mean={:.3f} min={:.3f} max={:.3f}".format(
                roi_name, stats["mean"], stats["min"], stats["max"]
            )
        )
        return stats

    def obstacle_detected_frame(self, frame):
        if self.config.get("avoidance.strategy", "disabled") == "disabled":
            return False
        stats = self.observe_frame("obstacle_depth_roi", frame)
        return bool(stats and stats["mean"] < float(self.avoidance_settings.get("obstacle_depth", 1.2)))
