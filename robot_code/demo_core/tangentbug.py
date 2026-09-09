from __future__ import print_function

import cv2
import numpy as np


class ObstacleAnalysis(object):
    """Serializable geometry plus array data used by the tuning HUD."""

    def __init__(
        self,
        detected=False,
        path_blocked=False,
        unsafe=False,
        contour=None,
        bbox=None,
        center=None,
        mask=None,
        background_depth=None,
        obstacle_depth=None,
        left_clearance_norm=0.0,
        right_clearance_norm=0.0,
        selected_side=None,
        reason=None,
        status="CLEAR",
        center_depth=None,
        candidate_bboxes=None,
    ):
        self.detected = bool(detected)
        self.path_blocked = bool(path_blocked)
        self.unsafe = bool(unsafe)
        self.contour = contour
        self.bbox = bbox
        self.center = center
        self.mask = mask
        self.background_depth = background_depth
        self.obstacle_depth = obstacle_depth
        self.left_clearance_norm = float(left_clearance_norm)
        self.right_clearance_norm = float(right_clearance_norm)
        self.selected_side = selected_side
        self.reason = reason
        self.status = str(status)
        self.center_depth = center_depth
        self.candidate_bboxes = list(candidate_bboxes or [])

    def as_dict(self):
        bbox = None
        width_norm = None
        height_norm = None
        center_x_norm = None
        bbox_left_norm = None
        bbox_right_norm = None
        if self.bbox is not None and self.mask is not None:
            x, y, width, height = self.bbox
            image_height, image_width = self.mask.shape[:2]
            bbox = [int(x), int(y), int(width), int(height)]
            width_norm = float(width) / float(max(1, image_width))
            height_norm = float(height) / float(max(1, image_height))
            bbox_left_norm = float(x) / float(max(1, image_width))
            bbox_right_norm = float(x + width) / float(max(1, image_width))
            if self.center is not None:
                center_x_norm = float(self.center[0]) / float(max(1, image_width))
        return {
            "detected": self.detected,
            "path_blocked": self.path_blocked,
            "unsafe": self.unsafe,
            "bbox": bbox,
            "center": list(self.center) if self.center is not None else None,
            "width_norm": width_norm,
            "height_norm": height_norm,
            "center_x_norm": center_x_norm,
            "bbox_left_norm": bbox_left_norm,
            "bbox_right_norm": bbox_right_norm,
            "background_depth": self.background_depth,
            "obstacle_depth": self.obstacle_depth,
            "left_clearance_norm": self.left_clearance_norm,
            "right_clearance_norm": self.right_clearance_norm,
            "selected_side": self.selected_side,
            "reason": self.reason,
            "status": self.status,
            "center_depth": self.center_depth,
            "candidate_bboxes": [list(item) for item in self.candidate_bboxes],
        }


class TangentPlan(object):
    def __init__(self, action, heading_error=0.0, path_clear=False, candidates=None, profile=None, reason=None):
        self.action = action
        self.heading_error = float(heading_error)
        self.path_clear = bool(path_clear)
        self.candidates = candidates or []
        self.profile = profile
        self.reason = reason

    def as_dict(self):
        return {
            "action": self.action,
            "heading_error": self.heading_error,
            "path_clear": self.path_clear,
            "candidates": self.candidates,
            "reason": self.reason,
        }


class DepthTangentBugPlanner(object):
    """TangentBug-inspired local planner for a monocular depth field.

    The original TangentBug assumes metric range sensing and robot pose. This
    adapter builds a one-dimensional local tangent graph from horizontal depth
    discontinuities. It is a reactive demo planner and does not claim the
    original algorithm's global convergence guarantee.
    """

    def __init__(self, settings):
        self.settings = settings

    def _column_profile(self, depth_map):
        depth = np.asarray(depth_map, dtype=np.float32)
        if depth.ndim > 2:
            depth = depth[:, :, 0]
        height, width = depth.shape[:2]
        y1 = max(0, min(height - 1, int(height * float(self.settings.get("profile_row_start", 0.45)))))
        y2 = max(y1 + 1, min(height, int(height * float(self.settings.get("profile_row_end", 0.9)))))
        columns = max(8, int(self.settings.get("profile_columns", 80)))
        edges = np.linspace(0, width, columns + 1).astype(np.int32)
        profile = np.full(columns, np.inf, dtype=np.float32)
        band = depth[y1:y2]
        for index in range(columns):
            region = band[:, edges[index]:edges[index + 1]]
            values = region[np.isfinite(region) & (region > 0.0)]
            if values.size:
                profile[index] = float(np.median(values))
        window = max(1, int(self.settings.get("smoothing_window", 5)))
        if window > 1:
            kernel = np.ones(window, dtype=np.float32) / float(window)
            finite_cap = float(self.settings.get("clear_depth", 1.6)) * 2.0
            smooth_input = np.where(np.isfinite(profile), profile, finite_cap)
            profile = np.convolve(smooth_input, kernel, mode="same").astype(np.float32)
        return profile

    @staticmethod
    def _find_contours(mask):
        result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return result[0] if len(result) == 2 else result[1]

    def analyze_obstacle(self, depth_map):
        """Classify center-path depth geometry without assuming an object class."""
        if depth_map is None:
            return ObstacleAnalysis(unsafe=True, path_blocked=True, status="BLOCKED", reason="depth unavailable")
        depth = np.asarray(depth_map, dtype=np.float32)
        if depth.ndim > 2:
            depth = depth[:, :, 0]
        if depth.ndim != 2 or not depth.size:
            return ObstacleAnalysis(unsafe=True, path_blocked=True, status="BLOCKED", reason="invalid depth shape")

        height, width = depth.shape[:2]
        row_start = float(self.settings.get("contour_row_start", self.settings.get("profile_row_start", 0.45)))
        row_end = float(self.settings.get("contour_row_end", self.settings.get("profile_row_end", 0.9)))
        y1 = max(0, min(height - 1, int(round(height * row_start))))
        y2 = max(y1 + 1, min(height, int(round(height * row_end))))
        side_fraction = min(0.45, max(0.05, float(self.settings.get("background_side_fraction", 0.2))))
        side_width = max(1, int(round(width * side_fraction)))
        outer = np.concatenate((depth[y1:y2, :side_width].ravel(), depth[y1:y2, width - side_width:].ravel()))
        outer = outer[np.isfinite(outer) & (outer > 0.0)]
        if not outer.size:
            return ObstacleAnalysis(unsafe=True, path_blocked=True, status="BLOCKED", reason="background depth unavailable")
        background_depth = float(np.median(outer))
        minimum_drop = max(0.0, float(self.settings.get("relative_depth_drop", 0.25)))
        maximum_depth = float(self.settings.get("maximum_obstacle_depth", background_depth))
        cutoff = min(background_depth - minimum_drop, maximum_depth)

        valid = np.isfinite(depth) & (depth > 0.0)
        mask = np.zeros((height, width), dtype=np.uint8)
        top_ignore_ratio = min(0.9, max(0.0, float(self.settings.get("obstacle_top_ignore_ratio", 0.0))))
        bottom_ignore_ratio = min(0.9, max(0.0, float(self.settings.get("obstacle_bottom_ignore_ratio", 0.0))))
        obstacle_y1 = max(y1, min(height - 1, int(round(height * top_ignore_ratio))))
        obstacle_y2 = min(y2, max(obstacle_y1 + 1, int(round(height * (1.0 - bottom_ignore_ratio)))))
        mask[obstacle_y1:obstacle_y2] = np.where(
            valid[obstacle_y1:obstacle_y2] & (depth[obstacle_y1:obstacle_y2] <= cutoff), 255, 0
        ).astype(np.uint8)
        kernel_size = max(1, int(self.settings.get("morphology_kernel", 5)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        corridor_width = min(0.9, max(0.02, float(self.settings.get("center_corridor_width_norm", 0.24))))
        corridor_left = width * (0.5 - corridor_width / 2.0)
        corridor_right = width * (0.5 + corridor_width / 2.0)
        corridor_x1 = max(0, int(round(corridor_left)))
        corridor_x2 = min(width, max(corridor_x1 + 1, int(round(corridor_right))))
        center_values = depth[y1:y2, corridor_x1:corridor_x2]
        center_values = center_values[np.isfinite(center_values) & (center_values > 0.0)]
        center_percentile = min(100.0, max(0.0, float(self.settings.get("center_depth_percentile", 20.0))))
        center_depth = float(np.percentile(center_values, center_percentile)) if center_values.size else None
        absolute_stop_depth = float(self.settings.get("absolute_center_stop_depth", 0.0))
        absolute_center_blocked = (
            absolute_stop_depth > 0.0
            and center_depth is not None
            and center_depth <= absolute_stop_depth
        )
        contours = []
        central = []
        for contour in self._find_contours(mask):
            x, y, box_width, box_height = cv2.boundingRect(contour)
            intersects_corridor = not (x + box_width < corridor_left or x > corridor_right)
            area_norm = float(cv2.contourArea(contour)) / float(max(1, width * height))
            width_norm = float(box_width) / float(max(1, width))
            height_norm = float(box_height) / float(max(1, height))
            item = (contour, (x, y, box_width, box_height), area_norm, width_norm, height_norm, intersects_corridor)
            contours.append(item)
            if intersects_corridor:
                central.append(item)
        candidate_bboxes = [item[1] for item in contours]

        if absolute_center_blocked and not central:
            return ObstacleAnalysis(
                detected=True,
                path_blocked=True,
                unsafe=True,
                mask=mask,
                background_depth=background_depth,
                center_depth=center_depth,
                status="BLOCKED",
                reason="absolute center depth stop",
                candidate_bboxes=candidate_bboxes,
            )

        if not contours:
            return ObstacleAnalysis(
                mask=mask,
                background_depth=background_depth,
                center_depth=center_depth,
                status="CLEAR",
                reason="center path clear",
                candidate_bboxes=candidate_bboxes,
            )

        candidates = central if central else contours
        contour, bbox, _, _, _, path_blocked = max(candidates, key=lambda item: item[2])
        x, y, box_width, box_height = bbox
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        values = depth[(contour_mask > 0) & valid]
        obstacle_depth = float(np.median(values)) if values.size else None
        center = (int(round(x + box_width / 2.0)), int(round(y + box_height / 2.0)))
        left_clearance = float(x) / float(max(1, width))
        right_clearance = float(width - (x + box_width)) / float(max(1, width))
        tie_margin = max(0.0, float(self.settings.get("side_selection_tie_norm", 0.03)))
        if abs(left_clearance - right_clearance) <= tie_margin:
            selected_side = str(self.settings.get("default_side", "left"))
        else:
            selected_side = "left" if left_clearance > right_clearance else "right"
        wall_width_threshold = min(
            1.0, max(0.0, float(self.settings.get("wall_width_threshold_norm", 0.6)))
        )
        wide_center_wall = (
            wall_width_threshold > 0.0
            and path_blocked
            and float(box_width) / float(max(1, width)) >= wall_width_threshold
        )
        if wide_center_wall:
            if absolute_center_blocked:
                return ObstacleAnalysis(
                    detected=True,
                    path_blocked=True,
                    unsafe=True,
                    contour=contour,
                    bbox=bbox,
                    center=center,
                    mask=mask,
                    background_depth=background_depth,
                    obstacle_depth=obstacle_depth,
                    center_depth=center_depth,
                    left_clearance_norm=left_clearance,
                    right_clearance_norm=right_clearance,
                    selected_side=None,
                    status="BLOCKED",
                    reason="wide wall reached absolute center stop",
                    candidate_bboxes=candidate_bboxes,
                )
            return ObstacleAnalysis(
                detected=False,
                path_blocked=False,
                unsafe=False,
                contour=contour,
                bbox=bbox,
                center=center,
                mask=mask,
                background_depth=background_depth,
                obstacle_depth=obstacle_depth,
                center_depth=center_depth,
                left_clearance_norm=left_clearance,
                right_clearance_norm=right_clearance,
                selected_side=None,
                status="IGNORED_WALL",
                reason="center bbox exceeds wall width threshold",
                candidate_bboxes=candidate_bboxes,
            )
        if path_blocked:
            status = "CANDIDATE"
            reason = "center bbox awaiting confirmation"
        else:
            status = "OBSTACLE"
            reason = "bbox outside center corridor"
        return ObstacleAnalysis(
            detected=True,
            path_blocked=path_blocked,
            unsafe=False,
            contour=contour,
            bbox=bbox,
            center=center,
            mask=mask,
            background_depth=background_depth,
            obstacle_depth=obstacle_depth,
            center_depth=center_depth,
            left_clearance_norm=left_clearance,
            right_clearance_norm=right_clearance,
            selected_side=selected_side,
            status=status,
            reason=reason,
            candidate_bboxes=candidate_bboxes,
        )

    @staticmethod
    def _segments(mask):
        segments = []
        start = None
        for index, value in enumerate(mask):
            if value and start is None:
                start = index
            if start is not None and (not value or index == len(mask) - 1):
                end = index if value and index == len(mask) - 1 else index - 1
                segments.append((start, end))
                start = None
        return segments

    def plan(self, depth_map, target_error_x=0.0):
        if depth_map is None:
            return TangentPlan("blocked", reason="depth unavailable")
        profile = self._column_profile(depth_map)
        count = len(profile)
        target_index = int(round((0.5 + float(target_error_x)) * float(count - 1)))
        target_index = max(0, min(count - 1, target_index))
        obstacle_depth = float(self.settings.get("obstacle_depth", 1.2))
        clear_depth = float(self.settings.get("clear_depth", 1.6))
        minimum_gap = max(1, int(self.settings.get("minimum_gap_columns", 6)))
        half_corridor = max(1, minimum_gap // 2)
        direct_start = max(0, target_index - half_corridor)
        direct_end = min(count, target_index + half_corridor + 1)
        direct_clear = bool(np.all(profile[direct_start:direct_end] >= clear_depth))
        if direct_clear:
            return TangentPlan(
                "forward",
                heading_error=float(target_error_x),
                path_clear=True,
                profile=profile,
                reason="direct corridor clear",
            )

        free = profile >= obstacle_depth
        segments = [segment for segment in self._segments(free) if segment[1] - segment[0] + 1 >= minimum_gap]
        if not segments:
            return TangentPlan("blocked", profile=profile, reason="no traversable depth gap")

        discontinuity = float(self.settings.get("discontinuity_threshold", 0.3))
        heading_weight = float(self.settings.get("heading_weight", 1.0))
        clearance_weight = float(self.settings.get("clearance_weight", 0.35))
        candidates = []
        for start, end in segments:
            indices = {start, end, int(round((start + end) / 2.0))}
            for index in sorted(indices):
                heading_error = (float(index) / float(max(1, count - 1))) - 0.5
                clearance = float(profile[index])
                tangent_bonus = 0.0
                if index > 0 and abs(float(profile[index]) - float(profile[index - 1])) >= discontinuity:
                    tangent_bonus = -0.1
                if index + 1 < count and abs(float(profile[index + 1]) - float(profile[index])) >= discontinuity:
                    tangent_bonus = -0.1
                cost = (
                    heading_weight * abs(index - target_index) / float(count)
                    + clearance_weight / max(clearance, 0.05)
                    + tangent_bonus
                )
                candidates.append({
                    "index": int(index),
                    "heading_error": float(heading_error),
                    "clearance": clearance,
                    "cost": float(cost),
                })
        chosen = min(candidates, key=lambda item: item["cost"])
        heading_error = float(chosen["heading_error"])
        if abs(heading_error) <= 0.08:
            action = "forward"
        else:
            action = "right" if heading_error > 0.0 else "left"
        return TangentPlan(
            action,
            heading_error=heading_error,
            path_clear=False,
            candidates=candidates,
            profile=profile,
            reason="local tangent candidate",
        )

    def draw_debug(self, frame, plan):
        canvas = np.asarray(frame).copy()
        height, width = canvas.shape[:2]
        profile = plan.profile
        if profile is None or len(profile) == 0:
            cv2.putText(canvas, "TangentBug: {}".format(plan.reason), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            return canvas
        max_depth = max(float(self.settings.get("clear_depth", 1.6)) * 1.5, 0.1)
        points = []
        for index, value in enumerate(profile):
            x = int(round(float(index) * float(width - 1) / float(max(1, len(profile) - 1))))
            normalized = min(max(float(value), 0.0), max_depth) / max_depth
            y = int(round(float(height - 1) - normalized * float(height) * 0.35))
            points.append([x, y])
            color = (0, 255, 0) if value >= float(self.settings.get("obstacle_depth", 1.2)) else (0, 0, 255)
            cv2.line(canvas, (x, height - 1), (x, y), color, 1)
        if len(points) > 1:
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, (255, 255, 0), 1)
        for candidate in plan.candidates:
            x = int(round(float(candidate["index"]) * float(width - 1) / float(max(1, len(profile) - 1))))
            cv2.circle(canvas, (x, height - 8), 4, (255, 0, 255), -1)
        cv2.putText(
            canvas,
            "TangentBug action={} clear={} err={:.3f}".format(plan.action, plan.path_clear, plan.heading_error),
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )
        return canvas

    def draw_obstacle_debug(
        self,
        frame,
        analysis,
        state="READY",
        recording=False,
        draw_geometry=True,
        draw_text=True,
    ):
        canvas = np.asarray(frame).copy()
        height, width = canvas.shape[:2]
        if draw_geometry and analysis.mask is not None:
            resized_mask = cv2.resize(analysis.mask, (width, height), interpolation=cv2.INTER_NEAREST)
            overlay = canvas.copy()
            overlay[resized_mask > 0] = (0, 0, 255)
            canvas = cv2.addWeighted(canvas, 0.72, overlay, 0.28, 0.0)
        depth_height, depth_width = analysis.mask.shape[:2] if analysis.mask is not None else (height, width)
        scale_x = float(width) / float(max(1, depth_width))
        scale_y = float(height) / float(max(1, depth_height))
        if draw_geometry and analysis.contour is not None:
            contour = np.asarray(analysis.contour, dtype=np.float32).copy()
            contour[:, :, 0] *= scale_x
            contour[:, :, 1] *= scale_y
            cv2.drawContours(canvas, [contour.astype(np.int32)], -1, (0, 255, 255), 2)
        corridor = float(self.settings.get("center_corridor_width_norm", 0.24))
        left = int(round(width * (0.5 - corridor / 2.0)))
        right = int(round(width * (0.5 + corridor / 2.0)))
        if draw_geometry:
            cv2.rectangle(canvas, (left, 0), (right, height - 1), (255, 180, 0), 2)
        top_ignore = min(0.9, max(0.0, float(self.settings.get("obstacle_top_ignore_ratio", 0.0))))
        bottom_ignore = min(0.9, max(0.0, float(self.settings.get("obstacle_bottom_ignore_ratio", 0.0))))
        top_y = int(round(height * top_ignore))
        bottom_y = int(round(height * (1.0 - bottom_ignore)))
        if draw_geometry and top_y > 0:
            cv2.line(canvas, (0, top_y), (width - 1, top_y), (0, 165, 255), 2)
            cv2.putText(
                canvas, "ROI TOP", (8, max(16, top_y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA,
            )
        if draw_geometry and bottom_ignore > 0.0:
            bottom_y = min(height - 1, max(0, bottom_y))
            cv2.line(canvas, (0, bottom_y), (width - 1, bottom_y), (0, 165, 255), 2)
            cv2.putText(
                canvas, "ROI BOTTOM", (8, max(16, bottom_y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA,
            )
        if draw_geometry and analysis.bbox is not None:
            x, y, box_width, box_height = analysis.bbox
            x1 = int(round(x * scale_x))
            x2 = int(round((x + box_width) * scale_x))
            y1 = int(round(y * scale_y))
            y2 = int(round((y + box_height) * scale_y))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
            margin = float(self.settings.get("tangent_margin_norm", 0.04))
            left_tangent = max(0, int(round(x1 - width * margin)))
            right_tangent = min(width - 1, int(round(x2 + width * margin)))
            cv2.line(canvas, (left_tangent, 0), (left_tangent, height - 1), (255, 0, 255), 2)
            cv2.line(canvas, (right_tangent, 0), (right_tangent, height - 1), (255, 0, 255), 2)
        if draw_text:
            details = analysis.as_dict()
            lines = [
                "STATE={} STATUS={}".format(state, analysis.status),
                "center={} background={}".format(
                    _format_number(analysis.center_depth),
                    _format_number(analysis.background_depth),
                ),
                "obstacle={} width={}".format(
                    _format_number(analysis.obstacle_depth),
                    _format_number(details.get("width_norm")),
                ),
            ]
            for index, text in enumerate(lines):
                y = 24 + index * 24
                cv2.rectangle(canvas, (5, y - 18), (min(width - 1, 8 + len(text) * 9), y + 5), (20, 20, 20), -1)
                cv2.putText(canvas, text, (9, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        if recording:
            cv2.circle(canvas, (width - 24, 22), 8, (0, 0, 255), -1)
            cv2.putText(canvas, "REC", (width - 76, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return canvas


def _format_number(value):
    return "n/a" if value is None else "{:.3f}".format(float(value))
