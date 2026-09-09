import unittest

import numpy as np

from demo_core.tangentbug import DepthTangentBugPlanner


SETTINGS = {
    "obstacle_depth": 1.2,
    "clear_depth": 1.6,
    "profile_columns": 40,
    "profile_row_start": 0.4,
    "profile_row_end": 0.9,
    "smoothing_window": 3,
    "minimum_gap_columns": 4,
    "absolute_center_stop_depth": 1.2,
    "wall_width_threshold_norm": 0.6,
}


class TangentBugTest(unittest.TestCase):
    def test_clear_center_returns_path_clear(self):
        depth = np.full((60, 80), 2.0, dtype=np.float32)
        plan = DepthTangentBugPlanner(SETTINGS).plan(depth)
        self.assertTrue(plan.path_clear)
        self.assertEqual(plan.action, "forward")

    def test_center_obstacle_selects_side_candidate(self):
        depth = np.full((60, 80), 2.0, dtype=np.float32)
        depth[25:55, 30:50] = 0.8
        plan = DepthTangentBugPlanner(SETTINGS).plan(depth)
        self.assertFalse(plan.path_clear)
        self.assertIn(plan.action, ("left", "right", "forward"))
        self.assertTrue(plan.candidates)

    def test_debug_overlay_keeps_frame_shape(self):
        depth = np.full((60, 80), 2.0, dtype=np.float32)
        planner = DepthTangentBugPlanner(SETTINGS)
        plan = planner.plan(depth)
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        self.assertEqual(planner.draw_debug(frame, plan).shape, frame.shape)

    def test_center_obstacle_contour_blocks_path(self):
        depth = np.full((80, 100), 2.0, dtype=np.float32)
        depth[30:72, 43:57] = 0.9
        planner = DepthTangentBugPlanner(SETTINGS)
        analysis = planner.analyze_obstacle(depth)
        self.assertTrue(analysis.detected)
        self.assertTrue(analysis.path_blocked)
        self.assertFalse(analysis.unsafe)
        self.assertEqual(analysis.status, "CANDIDATE")
        self.assertEqual(analysis.selected_side, "left")
        self.assertLess(analysis.obstacle_depth, analysis.background_depth)

    def test_off_center_obstacle_is_visible_but_does_not_block_path(self):
        depth = np.full((80, 100), 2.0, dtype=np.float32)
        depth[30:72, 72:84] = 0.9
        analysis = DepthTangentBugPlanner(SETTINGS).analyze_obstacle(depth)
        self.assertTrue(analysis.detected)
        self.assertFalse(analysis.path_blocked)

    def test_wide_center_obstacle_selects_a_side_without_blocking(self):
        depth = np.full((80, 100), 2.0, dtype=np.float32)
        depth[30:72, 25:75] = 0.9
        analysis = DepthTangentBugPlanner(SETTINGS).analyze_obstacle(depth)
        self.assertTrue(analysis.detected)
        self.assertFalse(analysis.unsafe)
        self.assertTrue(analysis.path_blocked)
        self.assertEqual(analysis.status, "CANDIDATE")
        self.assertIn(analysis.selected_side, ("left", "right"))
        self.assertEqual(analysis.reason, "center bbox awaiting confirmation")

    def test_wide_center_bbox_is_ignored_as_wall_until_close(self):
        depth = np.full((80, 100), 2.0, dtype=np.float32)
        depth[30:72, 15:85] = 1.4
        analysis = DepthTangentBugPlanner(SETTINGS).analyze_obstacle(depth)
        self.assertFalse(analysis.detected)
        self.assertFalse(analysis.path_blocked)
        self.assertFalse(analysis.unsafe)
        self.assertEqual(analysis.status, "IGNORED_WALL")
        self.assertEqual(analysis.reason, "center bbox exceeds wall width threshold")
        self.assertIsNotNone(analysis.bbox)

    def test_wide_center_wall_still_uses_absolute_close_protection(self):
        depth = np.full((80, 100), 2.0, dtype=np.float32)
        depth[30:72, 15:85] = 0.9
        analysis = DepthTangentBugPlanner(SETTINGS).analyze_obstacle(depth)
        self.assertTrue(analysis.detected)
        self.assertTrue(analysis.path_blocked)
        self.assertTrue(analysis.unsafe)
        self.assertEqual(analysis.status, "BLOCKED")
        self.assertEqual(analysis.reason, "wide wall reached absolute center stop")

    def test_small_center_bbox_is_candidate_without_shape_filter(self):
        depth = np.full((80, 100), 2.0, dtype=np.float32)
        depth[35:41, 48:54] = 0.9
        analysis = DepthTangentBugPlanner(SETTINGS).analyze_obstacle(depth)
        self.assertTrue(analysis.detected)
        self.assertTrue(analysis.path_blocked)
        self.assertFalse(analysis.unsafe)
        self.assertEqual(analysis.status, "CANDIDATE")

    def test_close_wall_is_blocked_by_absolute_center_depth(self):
        depth = np.full((80, 100), 0.9, dtype=np.float32)
        analysis = DepthTangentBugPlanner(SETTINGS).analyze_obstacle(depth)
        self.assertTrue(analysis.path_blocked)
        self.assertTrue(analysis.unsafe)
        self.assertEqual(analysis.status, "BLOCKED")
        self.assertEqual(analysis.reason, "absolute center depth stop")

    def test_top_ignore_ratio_only_excludes_obstacle_contour(self):
        settings = dict(SETTINGS)
        settings["obstacle_top_ignore_ratio"] = 0.3
        depth = np.full((100, 100), 2.0, dtype=np.float32)
        depth[5:25, 43:57] = 0.9
        analysis = DepthTangentBugPlanner(settings).analyze_obstacle(depth)
        self.assertFalse(analysis.detected)
        self.assertFalse(analysis.path_blocked)
        self.assertEqual(analysis.mask.shape, depth.shape)
        self.assertAlmostEqual(analysis.background_depth, 2.0)

    def test_bottom_ignore_ratio_excludes_floor_contour(self):
        settings = dict(SETTINGS)
        settings["obstacle_bottom_ignore_ratio"] = 0.2
        depth = np.full((100, 100), 2.0, dtype=np.float32)
        depth[84:99, 43:57] = 0.9
        analysis = DepthTangentBugPlanner(settings).analyze_obstacle(depth)
        self.assertFalse(analysis.detected)
        self.assertFalse(analysis.path_blocked)
        self.assertEqual(analysis.mask.shape, depth.shape)
        self.assertAlmostEqual(analysis.background_depth, 2.0)

    def test_contour_hud_keeps_frame_shape(self):
        depth = np.full((80, 100), 2.0, dtype=np.float32)
        depth[30:72, 43:57] = 0.9
        planner = DepthTangentBugPlanner(SETTINGS)
        analysis = planner.analyze_obstacle(depth)
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        hud = planner.draw_obstacle_debug(frame, analysis, state="DETECTED", recording=True)
        self.assertEqual(hud.shape, frame.shape)

    def test_geometry_and_text_hud_layers_are_independent(self):
        depth = np.full((80, 100), 2.0, dtype=np.float32)
        depth[30:72, 43:57] = 0.9
        planner = DepthTangentBugPlanner(SETTINGS)
        analysis = planner.analyze_obstacle(depth)
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        plain = planner.draw_obstacle_debug(
            frame, analysis, draw_geometry=False, draw_text=False
        )
        geometry = planner.draw_obstacle_debug(
            frame, analysis, draw_geometry=True, draw_text=False
        )
        text = planner.draw_obstacle_debug(
            frame, analysis, draw_geometry=False, draw_text=True
        )
        self.assertTrue(np.array_equal(plain, frame))
        self.assertFalse(np.array_equal(geometry, frame))
        self.assertFalse(np.array_equal(text, frame))


if __name__ == "__main__":
    unittest.main()
