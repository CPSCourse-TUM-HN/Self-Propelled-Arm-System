# Parameter Quick Reference

This is a short operational reference for the current `vague_map_imp` demo. Values come from `empirical_parameters.json` unless a standalone tuning JSON is named.

## Base motion

| Parameter | Current | Meaning | Test |
| --- | ---: | --- | --- |
| `navigation.can/bin.search.routine` | `0` | Selects a low-priority path from `predefined_routines.json`; vision always runs first. | full demo dry-run |
| `predefined_routines.json` IDs `0 / 2 / 3 / 4` | rotate / 0.30 m square / square+spins / map-arrival 400° scan | Routines 2/3 share the configured side length; routine 4 is selected only after vague-map bin arrival without Tag detection. | searching routine unit tests / real-base observation |
| `navigation.bin.search.map_arrival_routine` | `4` | After reaching the mapped bin docking point without seeing the Tag, scan 360°+40° with vision active. | full demo dry-run |
| `base_motion_speed_profiles.linear.slow/fast` | `0.2 / 0.6` | Named JetBot linear command speeds. | `base_speed_time_calibration.ipynb` |
| `base_motion_speed_profiles.turn.slow/fast` | `0.15 / 0.3` | Named JetBot turn command speeds. | `base_speed_time_calibration.ipynb` |
| `base_motion_command_scales.forward/backward/left/right` | all `1.0` | Empirical per-direction command multiplier. `effective_speed = requested_speed * scale`; this is not a friction coefficient. | `base_speed_time_calibration.ipynb` |
| `vague_map.odometry.linear_response_samples` | `0.2 × 4 s = 0.13 m; 0.6 × 4 s = 0.415 m` | Speed-specific open-loop linear calibration; exact speeds use their sample and intermediate speeds interpolate the response coefficient. | `base_motion_performance.ipynb` |
| `vague_map.odometry.linear_meters_per_speed_second` | `0.171875` | Least-squares fallback coefficient for speeds outside the measured 0.2–0.6 range. | `base_motion_performance.ipynb` |
| `vague_map.odometry.linear_slip_factor` | `1.0` | Linear odometry-only empirical multiplier. | `base_motion_performance.ipynb` |
| `vague_map.odometry.angular_radians_per_speed_second` | `pi` | Open-loop angular response estimate. | both base notebooks |
| `base_turn_response.samples` | `0.15: L 10.4/R 9.0 s; 0.3: L 3.6/R 3.4 s` | Measured time for a 90-degree (`pi/2` radian) turn; exact-speed lookup overrides the linear angular fallback. | `base_motion_performance.ipynb` / `turn_params.csv` |

Use `surface_label` in calibration CSVs rather than treating surface effects as a known physical friction coefficient. The calibration tool reports response in degrees per `(effective speed * second)`, recommended 90-degree duration, and neutralized left/right scale suggestions.
Timed opposite turns in docking and avoidance are converted through the direction-specific response table so they target the same physical angle rather than blindly reusing seconds.

`vague_map_tester.ipynb` uses the standalone `tuning_tools/vague_map_test_parameters.json`. Its
`initial_pose`, `bin_marker_position`, and `bin_docking_pose` describe the test layout; navigation uses
the docking pose as its coarse destination. The forward calibration result is
`measured_distance_m / (effective_speed * seconds * linear_slip_factor)` and is reported as a suggested
`linear_meters_per_speed_second`; the Notebook never writes that value or the production
`linear_response_samples` into production parameters.

## Bin / AprilTag docking

| Parameter | Current | Meaning | Test |
| --- | ---: | --- | --- |
| `detectors.bin.marker_length_m` | `0.04 m` | Outer black Tag square side length used by PnP. | `apriltag_pnp_stability_test.ipynb` |
| `camera.calibration_yaml` / `camera.rectification_alpha` | `calibration_320.yaml` / `0.0` | Single shared calibration source for AprilTag PnP and optional frame rectification; alpha 0 keeps rectified 320x240 output free of black borders by slightly cropping the field of view. | camera/depth diagnostics |
| `camera.depth_rectification_enabled` | `false` | Optionally rectify frames before DepthNet, depth ROI sampling and obstacle extraction. Keep disabled until the depth model is validated or fine-tuned for rectified input. | obstacle / dual-obstacle notebooks |
| `detectors.can.rectification_enabled` | `false` | Optionally rectify DetectNet input and its matching HUD/recording space. Default remains raw to match can training images. | dual-can / can stable-stop notebooks |
| `vague_map.bin_marker_position.heading_rad` | `0 rad` | World heading of the Tag-frame `+Z` axis used by PnP localization. | `apriltag_pnp_stability_test.ipynb` / physical survey |
| `vague_map.bin_tag_localization.enabled` | `true` | Attempts final stable Tag pose correction; rejected observations retain the fixed docking-pose fallback. | bin docking tuning |
| `vague_map.bin_tag_localization.minimum/maximum_tag_distance_m` | `0.01 / 100.0 m` | Demonstration-wide accepted PnP distance interval. Tighten after mount geometry is calibrated. | PnP stability / bin docking |
| `vague_map.bin_tag_localization.maximum_position_error_m` / `maximum_heading_error_rad` | `666.6 m / pi rad` | Demonstration-wide pose-jump gates; they currently reject only extreme or invalid results. | bin docking tuning |
| `vague_map.bin_tag_localization.camera_mounts.front/side` | zero offsets; headings `0 / +pi/2` placeholders | Camera optical center and optical-forward heading in the robot base frame for each arm view. Must be calibrated before enabling. | bin docking tuning |
| `navigation.bin.align.tolerance_norm` | `0.08` | Front-view horizontal alignment tolerance. | `bin_docking_tuning.ipynb` |
| `navigation.bin.align.target_error_x_norm` | `-0.16` | Desired normalized Tag-center error for front alignment and final verification. Bin `near_align` inherits this value. | `bin_docking_tuning.ipynb` |
| `navigation.bin.approach.stop.pnp_z_threshold_m` | `0.36 m` PnP `z` | Single front-approach stop threshold: move forward while `z` is larger, then continue to final verification once `z <= threshold`. | `bin_docking_tuning.ipynb` |
| `navigation.bin.side_docking.experimental.center_tolerance_norm` | `0.04` | Side-view longitudinal centering tolerance. | `bin_docking_tuning.ipynb` |
| `navigation.bin.side_docking.experimental.yaw_tolerance_rad` | `0.1281 rad` (7.34 deg) | Accepted OpenCV Tag-plane left/right yaw error. | `bin_docking_tuning.ipynb` |
| `navigation.bin.side_docking.experimental.yaw_single_correction_max_seconds` | `999 s` | Maximum duration of one continuous yaw segment before stopping to recheck x; effectively inactive at the default value. | `bin_docking_tuning.ipynb` |
| `navigation.bin.side_docking.experimental.target_yaw_rad` | `0 rad` (0 deg) | Measured side-docking PnP yaw at physical alignment; control corrects the difference from this target. | `bin_docking_tuning.ipynb` |
| `navigation.bin.side_docking.experimental.center_stage_entry_yaw_rad` | `0.3445 rad` (19.7 deg) | In pulse mode, latch into cx correction once coarse yaw enters this range; finish cx before final yaw correction. | `bin_docking_tuning.ipynb` |
| `navigation.bin.side_docking.experimental.coarse_yaw_center_guard_norm` | `0.03` | Maximum cx deviation during yaw: pulse uses it in coarse yaw; continuous now stops yaw and recenters fully inside this guard before resuming. | `bin_docking_tuning.ipynb` |
| `navigation.bin.side_docking.experimental.yaw_turn_speed` / `yaw_positive_turn_direction` | `turn.slow` / `right` | Controls continuous signed yaw correction. Yaw is completed before continuous cx correction starts, then yaw is checked again. | `bin_docking_tuning.ipynb` / `base_motion_performance.ipynb` |
| `navigation.bin.side_docking.experimental.correction_mode` | `continuous` | `continuous` uses continuous staged correction; `pulse` performs coarse yaw, latched cx correction, then final yaw correction. | `bin_docking_tuning.ipynb` |
| `navigation.bin.side_docking.experimental.yaw_correction_seconds` / `center_correction_seconds` | `0.4 s / 0.3 s` | Pulse lengths used only when `correction_mode` is `pulse`. | `bin_docking_tuning.ipynb` |
| `navigation.bin.side_docking.experimental.side_entry_base_turn_direction` | `left` | After Servo 1 reaches its fixed side pose, the base makes the opposing calibrated 90° coarse turn; side-view Tag yaw/center then close the loop. | `bin_docking_tuning.ipynb` |
| `navigation.bin.side_docking.experimental.post_entry_camera_settle_seconds` / `post_entry_camera_discard_frames` | `0.4 s` / `3` | Ignore motion-blurred/cached camera data after the coarse base turn before starting Tag corrections. | `bin_docking_tuning.ipynb` |
| `navigation.bin.side_docking.experimental.post_correction_camera_settle_seconds` | `0.2 s` | Pause after one correction axis reaches tolerance before reading a fresh Tag observation for the next axis. | `bin_docking_tuning.ipynb` |
| `arm.poses.side_view_grabbing.angles.s1` | `70` | Fixed calibrated 90-degree side-view pose; Servo 1 is not adjusted during docking yaw correction. | `bin_docking_tuning.ipynb` |
| `arm.poses.side_view_bin_insert.angles` | `70, 80, 55, -55, 0` | Post-correction insertion pose. It is entered only after side docking is stable and keeps Servo 4 closed. | arm tuning / bin docking |
| `arm.poses.release.angles` | `70, 80, 55, 0, 0` | Same insertion geometry with only Servo 4 opened. | arm tuning / bin docking |
| `navigation.bin.side_docking.experimental.stable_frames` | `5` | Consecutive acceptable observations required. | `bin_docking_tuning.ipynb` |

The PnP stability notebook reports `x/y/z`, perpendicular distance, total distance, yaw/pitch/roll, detection rate, standard deviation and MAD. It deliberately does not fall back to DepthNet.

## Pickup and arm

| Parameter | Current | Meaning | Test |
| --- | ---: | --- | --- |
| `navigation.can.approach.stop.mode` | `bbox_height` | Selects `bbox_height`, center-ROI `depth`, or `bbox_height_and_depth` (AND). | `can_stable_stop.ipynb` |
| `navigation.can.approach.stop.bbox_height_threshold` | `0.36` | Stop-side bbox condition: normalized can box height must be at least this value. | `can_stable_stop.ipynb` |
| `navigation.can.approach.stop.depth_raw_threshold` | `2.05` (inactive in bbox-only mode) | Stop-side DepthNet lens-center raw condition: mean raw depth must be at most this value. | `can_stable_stop.ipynb` |
| `arm.speed` | `85` | General servo-controller speed. | `arm_sequence_tuning.ipynb` |
| `arm.push.speed` | `linear.slow` | Base speed used for the pre-grab push. | full integration / arm tuning |
| `arm.push.seconds` | `3.9 s` | Pre-grab base push duration. | full integration / arm tuning |
| `arm.position_wait.stability_delta_raw` | `3` | Maximum sample-to-sample raw position change considered stable. | `arm_grasp_telemetry_test.ipynb` |
| `arm.position_wait.stable_samples` | `2` | Consecutive stable position samples required. | arm telemetry notebook |
| `arm.poses.*.angles.s1..s5` | pose-specific | Saved joint and gripper targets. | `arm_sequence_tuning.ipynb` |

`arm_grasp_telemetry_test.ipynb` reads position, actual speed, load, bus voltage, temperature, moving status, current and communication errors. Treat load/current as experimental signals until empty-grab, can-grab and soft-block logs show repeatable separation.

## Depth obstacle extraction and bypass

Standalone values below come from `tuning_tools/obstacle_avoidance_parameters.json`.

| Parameter | Current | Meaning | Test |
| --- | ---: | --- | --- |
| `perception.relative_depth_drop` | `0.22` | Required relative foreground depth drop from estimated background. | obstacle / dual-obstacle notebooks |
| `perception.maximum_obstacle_depth` | `2.4` | Upper cap for the obstacle-depth threshold. | same |
| `perception.morphology_kernel` | `4` | Mask morphology kernel; large values can merge nearby objects. | `dual_obstacle_detection_test.ipynb` |
| `perception.center_corridor_width_norm` | `0.24` | Collision corridor width. | obstacle tuning |
| `perception.tangent_margin_norm` | `0.06` | Extra image-width margin beyond the collision corridor before tangent heading is accepted. | obstacle tuning |
| `perception.obstacle_top_ignore_ratio` | `0.30` | Top image band excluded from contour extraction. | same |
| `perception.obstacle_bottom_ignore_ratio` | `0.15` | Bottom/floor band excluded from contour extraction. | same |
| `control.obstacle_confirmation_frames` | `15` | Stationary frames required to lock the first obstacle. | obstacle tuning |
| `control.linear_obstacle_confirmation_frames` | `3` | Frames required to confirm another obstacle during a linear segment. | obstacle tuning |
| `motion.depth_to_segment_time_factor` | `4.0` | First lateral duration factor based on locked obstacle depth. | obstacle tuning |
| `motion.pass_depth_to_time_factor` | `1.5` | Independent forward passing-duration factor. | obstacle tuning |

For dual-obstacle testing, use scenario labels `clearly_separated`, `nearly_touching`, `similar_depth`, and `front_back_offset`. Review expected-count rate, merge rate, false-split rate and the numbered candidate HUD before changing morphology or depth thresholds.

## Safe order

1. Run camera/network diagnostics.
2. Calibrate base left/right behavior on each surface.
3. Calibrate camera intrinsics, then measure PnP stability.
4. Tune arm poses; collect telemetry without changing pickup logic.
5. Test dual-obstacle extraction with the base disabled.
6. Run bin docking and finally the full integration notebook.
