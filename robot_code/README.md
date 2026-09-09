# JetTank Can Pickup Demo

This directory is the self-contained delivery package for the JetTank demo. With a browser available, use the JupyterLab notebooks as the primary operation, tuning, live-HUD, and recording interface. The same production state machine also has a Python command-line entry point for browserless or SSH-only operation. Three JSON files hold the active configuration.

## Runtime Flow

```text
INIT -> PLAN -> SEARCH CAN (selected routine; restart after timeout)
     -> ALIGN -> APPROACH -> FINAL_VERIFY -> PICKUP
     -> MAP NAVIGATE TO BIN -> visual ALIGN/PNP Z APPROACH -> optional exp2 SIDE DOCK -> RELEASE
     -> nearest mapped can or restart SEARCH -> DONE at pickup limit
```

The vague map uses command-based open-loop odometry to move toward approximate areas. Camera detection is attempted before every map movement update and immediately takes control when the target is visible. `APPROACH` can independently be interrupted by optional avoidance; the map itself does not plan around obstacles. Every terminal path calls `stop_all()`.

## Configuration

- `config.json`: project paths, dry-run switches, feature switches, pickup limit, retry policy, and avoidance strategy.
- `empirical_parameters.json`: camera dimensions, model settings, thresholds, named base-speed profiles, pulse lengths, ROIs, and arm poses.
- `predefined_routines.json`: declarative low-priority search paths selected by can/bin navigation settings.

DepthNet input can be rectified with `camera.calibration_yaml`; it is disabled by default. The depth switch
is `camera.depth_rectification_enabled`; Can DetectNet has an independent
`detectors.can.rectification_enabled` switch that defaults off. AprilTag remains on the
raw frame because calibrated PnP already consumes the distortion coefficients.

The merge order is:

```text
empirical_parameters.json -> config.json -> command-line overrides
```

## Recommended Workflow: JupyterLab With a Browser

The browser-based JupyterLab interface is the recommended way to operate, tune, and demonstrate the system. It exposes live camera/HUD output, recording controls, subsystem hardware switches, JSON reload controls, and cooperative stop actions without requiring command-line arguments.

From the `robot_code` directory on the Jetson, start JupyterLab:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Open the URL printed by JupyterLab in a browser on the same network. When working directly on a desktop-equipped Jetson, `jupyter lab` may be used to open the local browser automatically. Keep the terminal that launched JupyterLab available so the server can be stopped cleanly.

Use this Notebook sequence for a new board or demo setup:

1. Open `tuning_tools/camera_network_diagnostics.ipynb` to verify the camera, Can model, AprilTag detector, DepthNet, overlays, and model reuse without starting the complete mission.
2. Use the focused tuning notebooks only for the subsystem that needs adjustment: base motion, arm poses, avoidance, or bin docking.
3. Open `tests/full_demo_integration_test.ipynb` for the production `DemoStateMachine`, preflight checks, live HUD, optional recording, and the one-click complete run.

The main browser tools are:

- `tests/full_demo_integration_test.ipynb`: primary integration and demonstration interface.
- `tuning_tools/camera_network_diagnostics.ipynb`: camera and perception health check.
- `tuning_tools/camera_depthnet_interrupt_test.ipynb`: independent camera/DepthNet lifecycle and interruption check.
- `tuning_tools/base_motion_performance.ipynb`: manual motion, square/semicircle paths, speed switching, turn calibration, and odometry checks.
- `tuning_tools/base_speed_time_calibration.ipynb`: focused speed-times-duration calibration with responsive stop and CSV output.
- `tuning_tools/arm_sequence_tuning.ipynb`: arm-pose inspection and deliberate editing of the `arm` section in `empirical_parameters.json`.
- `tuning_tools/bin_docking_tuning.ipynb`: focused Tag search, approach, side-docking correction, optional insertion/release, live HUD, and recording.
- `tuning_tools/obstacle_avoidance_tuning.ipynb`: standalone DepthNet contour and simplified-TangentBug testing with live HUD and recording.

Before running a hardware cell, review the Notebook's camera, base, arm, live, and recording controls. Do not rely on a previously saved checkbox state. Leave motion disabled until the tracks are lifted or the floor area is clear, then enable only the hardware needed for that test. Recording remains an explicit choice because video output can consume storage and browser resources.

Notebook numeric controls use direct text-entry fields. Reload the relevant JSON parameters before a run when values were edited outside the Notebook. Detailed logs and recordings are written under the corresponding `logs/<tool>/<timestamp>/` directory; the UI intentionally presents the live HUD and a smaller operational status view.

Notebook Stop All is cooperative. It stops the base immediately, cancels arm polling, requests media shutdown, and returns control to the UI. The background worker finalizes any recording and owns `safe_home` and camera cleanup. Wait for `BACKGROUND RUN FINISHED status=STOPPED` before resetting the runtime or starting another run.

`exp2` in the full-demo Notebook enables right-facing bin docking. The arm settles at `side_view_grabbing`, the base performs its direction-specific calibrated 90-degree entry turn, and Servo 1 remains fixed in the side-view pose. AprilTag PnP yaw corrections are then performed by the base. A stable result moves through `side_view_bin_insert`, opens Servo 4 without changing insertion geometry, returns to `safe_home`, and records the configured side-parking pose.

The obstacle-avoidance Notebook reads only `tuning_tools/obstacle_avoidance_parameters.json`; it does not modify production empirical parameters. `Start Live` is a stationary monitor. Starting an analysis or bypass job replaces that monitor, while the `live` checkbox keeps previews enabled during motion.

## Command-Line Workflow Without a Browser

Use the Python entry points when JupyterLab or a browser is unavailable, such as a serial console or SSH-only session. From the `robot_code` directory:

```bash
python3 scripts/board_first_checks.py
python3 scripts/home_pipeline_smoke.py
python3 run_demo.py --validate-only --no-log-file
python3 run_demo.py --dry-run --no-log-file
```

`home_pipeline_smoke.py` is the short, motion-safe pre-demo check. It initializes the real camera and DepthNet once, invokes the Can and AprilTag detectors, releases their handles, and then runs the FSM in full dry-run. Can/Tag `found=false` is a warning rather than a failure. Use `--simulation-only` away from the Jetson.

Real hardware can be enabled together or independently:

```bash
python3 run_demo.py --real
python3 run_demo.py --camera-real --base-real
python3 run_demo.py --camera-real --base-real --arm-real --avoidance tangentbug_depth
```

For focused command-line base calibration:

```bash
python3 tuning_tools/base_speed_time_calibration.py --list
python3 tuning_tools/base_speed_time_calibration.py --case forward_slow_1s --real
python3 tuning_tools/base_speed_time_calibration.py --case turn_left_slow_1s --real
```

Running multiple real calibration cases requires `--confirm-multiple`. `Ctrl+C` and every normal/error path call `base.stop()`. Hardware is dry-run unless explicitly enabled.

Normal straight approach and vague-map travel use continuous constant-speed commands. Vision, target-loss limits, arrival thresholds, and state timeouts constrain and stop those commands. Search, target alignment, approach steering, map heading correction, and TangentBug feedback turns are continuous. Side docking corrects one axis at a time from calibrated AprilTag PnP yaw and normalized Tag-center error; `continuous` and fixed-duration `pulse` modes remain available. Scripted avoidance remains time-bounded. The Notebook media thread only renders camera frames and cached observations and never invokes a detector or DepthNet itself.

Base movement speeds are selected symbolically from `base_motion_speed_profiles`: `linear.fast=0.6`, `linear.slow=0.2`, `turn.fast=0.3`, and `turn.slow=0.15`. Motion settings reference names such as `linear.fast` rather than duplicating numeric speeds. Servo speeds remain independent integer parameters because they use a different controller and unit.

## Vague Map

`vague_map.enabled=true` enables a runtime-only 2D map. The manually placed start pose is `(0,0,0)`: `+Y` is forward, `+X` is right, and positive heading rotates clockwise from `+Y` toward `+X`. Units are meters and radians. After right-side docking, a bin directly beside the robot is therefore `(a,0)` in robot-relative coordinates, with `a > 0`. The configured bin and docking poses are examples and must be measured for the real demo area.

`SEARCHING` uses a predefined low-priority motion routine from `predefined_routines.json`; vision is evaluated before every routine update and always preempts base motion. Routine 0 preserves the simple continuous rotating search. Routine 2 traces a square from the left-edge midpoint, and routine 3 adds open-loop 360-degree turns at the lower and upper edge midpoints. Select a routine through `navigation.can.search.routine` or `navigation.bin.search.routine`.

Routines 2 and 3 are obstacle-aware when avoidance is enabled. Before a translation command, depth detection may pause the routine and enter `AVOIDING`. The executor retains completed theoretical distance, records the avoidance odometry delta, resumes the interrupted sequence, and adds a coarse position/heading return to the cycle origin after a displaced cycle. The resulting path may be rectangular or otherwise deformed; preserving the exact square is not a requirement. Routine 0 contains no translation and keeps its simpler behavior.

`vague_map.bin_tag_localization.enabled=true` currently attempts to correct the robot pose from the final stable AprilTag PnP observation. Its distance, position-jump, and heading-jump gates are intentionally very wide for the demonstration; rejected observations retain the fixed docking-pose fallback. Meaningful metric localization still requires surveyed Tag world pose and calibrated front/side camera-to-base transforms.

`runtime.max_pickups=1` is the default demo policy: after one successful release the next planning tick enters `DONE`. Set it to `0` for open-ended operation; after cached cans are exhausted, Planning starts the selected search routine again. Disable the map only with a positive pickup limit.

After a fixed or AprilTag-derived docking pose correction, all remembered `MappedCan.position` values receive the same planar rigid transform. This preserves their direction and distance relative to the corrected robot pose instead of leaving them in the pre-correction odometry frame.

The most important calibration values are under `vague_map.odometry` and `vague_map.navigation`. DepthNet estimates are multiplied by `depth_to_distance_scale_m_per_unit` only for approximate can coordinates; final alignment and stopping remain visual.

## Models

Can detection uses the standard jetson-inference `detectNet` API directly:

```text
assets/models/detectnet_native_can/can_ssd_mobilenet_v1.onnx
assets/models/detectnet_native_can/labels.txt
input_0 -> scores + boxes
confidence=0.20, clustering=0.30
```

Run the first Nano image smoke test before enabling base or arm motion:

```bash
python3 scripts/validate_can_detectnet_image.py assets/samples/can1.jpg \
  --output diagnostic_outputs/detectnet_native_can.jpg
```

There is no runtime backend switch or custom TFOD/PyCUDA post-processing path in this branch. Can parameters are stored directly under `detectors.can`.

AprilTag detection uses `pupil_apriltags` with `tag36h11`; it does not require a learned model. The printable tag is `assets/bin_apriltag_36h11_id_0.png`.

`tuning_tools/bin_docking_tuning.ipynb` provides a live Tag/state overlay, optional MJPG recording of that
overlay, individual docking-stage controls, and a one-click bin-only pipeline from search through release/reset.
Camera, base, and arm hardware remain independent opt-ins, and the full pipeline stops before starting another
pickup cycle. Each individual stage is independently runnable: it prepares a fresh `grabbed=True`, bin-targeted
runtime and assumes its preceding stages succeeded.

For a focused vague-map return-to-bin test, open `tuning_tools/vague_map_tester.ipynb`. Edit
`tuning_tools/vague_map_test_parameters.json`, reload it in the Notebook, and verify the predicted first
motion before enabling the base. The test forces the mission context to `grabbed=True`, proves that
Planning selects map navigation instead of normal search, and can deliberately hide the AprilTag so
the base continues toward the encoded bin docking point.

## Logs And References

Runtime logs are written under `logs/` and ignored by Git. See `PROJECT_STRUCTURE.md`, `FSM_ARCHITECTURE.md`, `VAGUE_MAP_NAMING.md`, and `tests/NOTEBOOK_PARAMETER_GUIDE.md` for detailed ownership and testing guidance.

`legacy_codes.zip` is an optional historical archive and is not required to run the current system. In particular,
`legacy_codes/legacy_params/useful_arm_states_record.json` preserves manually collected arm-pose references from
earlier tuning work: servo angles, command order, and the pause used between commands. It is useful for tracing how
named poses evolved, comparing current poses with known physical configurations, and recovering a conservative
starting point when retuning the arm.

The record is evidence and reference data, not an active configuration source. It contains retired and duplicate
state names, does not account for every later mechanical or calibration change, and is never loaded by production
code. Current arm poses in `empirical_parameters.json` remain authoritative; any legacy pose must be reviewed against
the current servo limits and hardware before it is commanded. Because the archive is large and binary, it should be
stored with Git LFS or as a separate release artifact rather than normal Git object storage.
