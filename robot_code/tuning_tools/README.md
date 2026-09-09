# Tuning Tools

This directory contains browser-based JupyterLab tools for calibrating, diagnosing, and demonstrating individual parts of the JetTank system. These tools reuse the production controllers and perception services, but they are not alternative production entry points. Run the complete mission from `tests/full_demo_integration_test.ipynb` or `run_demo.py`.

## Start Here

From the `robot_code` directory on the Jetson, start JupyterLab:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Open the URL printed by JupyterLab in a browser on the same network. The notebooks locate the project root from either the repository root or this directory.

Before enabling hardware:

1. Start with camera, base, and arm motion disabled unless the test explicitly requires them.
2. Lift the tracks or clear the floor before enabling `base_real`.
3. Keep only one notebook in control of a camera, servo port, or base at a time.
4. Use the notebook's STOP/Release action and wait for its background worker to finish before closing the kernel or opening another hardware notebook.
5. Treat recording as opt-in. AVI, CSV, images, and detailed logs can grow quickly.

A practical setup order is camera diagnostics, camera calibration/PnP verification, base calibration, arm tuning, can stopping, vague-map/bin docking, and finally obstacle tests.

## Notebook Index

| Notebook | Purpose | Hardware and motion | Persistent effect |
| --- | --- | --- | --- |
| `camera_network_diagnostics.ipynb` | First-line camera, Can DetectNet, AprilTag, DepthNet, overlay, and model-reuse check. | Camera only; no base or arm motion. | Optional captured images under `diagnostic_outputs/`. |
| `camera_depthnet_interrupt_test.ipynb` | Tests CameraOnly and DepthNet start, background model loading, interruption, unload, and release behavior. | Camera/DepthNet only. | Diagnostic output only. |
| `camera_intrinsic_calibration.ipynb` | Captures checkerboard views and calculates camera matrix, distortion, and reprojection error. | Camera only. | Writes captures under `logs/camera_calibration/`; **Save YAML** writes the selected calibration path but does not update production configuration. |
| `apriltag_pnp_stability_test.ipynb` | Measures multi-frame AprilTag translation, distance, and yaw stability with a selected calibration file. | Camera only. | CSV under `logs/apriltag_pnp/`; requires valid calibration and never falls back to DepthNet. |
| `base_motion_performance.ipynb` | Manual base commands, square and semicircle paths, speed switching, turn response, and odometry checks. | Real base motion is possible. | No production parameter write; results must be transferred deliberately. |
| `base_speed_time_calibration.ipynb` | Focused speed-times-duration trials with measured distance/angle entry and left/right 90-degree suggestions. | Real base motion is optional and disabled until selected. | Reads `base_speed_time_parameters.json`; writes/update-by-`run_id` CSV under `logs/base_speed_time/`. |
| `arm_grasp_telemetry_test.ipynb` | Reads SCSCL telemetry for servo IDs 1–5 and records load, voltage, current, temperature, position, speed, and movement state. | Real arm connection required; commanded motion is restricted to Servo 4 and requires explicit confirmation. | CSV under `logs/arm_telemetry/`; does not change grasp logic. |
| `arm_sequence_tuning.ipynb` | Inspects and tests pickup, carry, insertion, release, and safe-home poses. | Can command the arm; its push test can also move the base. | **Save Selected** is the only tuning action here that deliberately updates the `arm` section of `empirical_parameters.json`. |
| `can_stable_stop.ipynb` | Tunes the visually controlled stopping position immediately before `arm_down`, using bbox height, center depth, or their AND combination. | Camera and optional base motion; never runs the arm-down/grab sequence. | CSV and optional AVI under `logs/can_stable_stop/`; does not update production thresholds automatically. |
| `vague_map_tester.ipynb` | Starts with `grabbed=True` and verifies that coarse vague-map navigation toward the bin overrides normal searching. Also estimates a forward odometry factor. | Optional camera and base; no arm action and no visual docking sequence. | Reads `vague_map_test_parameters.json`; does not modify production parameters. |
| `bin_docking_tuning.ipynb` | Focused bin Tag search/approach and side-docking correction, with optional insert, release, and safe-home continuation. | Camera and base; arm movement occurs only when the selected flow includes release and arm hardware is enabled. | Detailed log plus optional synchronized AVI/CSV under `logs/bin_docking_tuning/`; reloads current production JSON rather than maintaining a duplicate parameter file. |
| `dual_can_vague_map_recording_test.ipynb` | Displays every visible Can bbox, optionally visualizes mapped candidates, and records evidence that additional cans can enter Vague Map while carrying. | Camera only; base and arm remain dry-run. | Optional AVI and CSV under `logs/dual_can_vague_map/`. |
| `dual_obstacle_detection_test.ipynb` | Tests whether DepthNet/TangentBug candidate extraction keeps two obstacles separate across frames. | Camera and DepthNet only; no base controller is constructed. | Optional AVI and CSV under `logs/dual_obstacle_detection/`. |
| `obstacle_avoidance_tuning.ipynb` | Full standalone simplified-TangentBug analysis and bypass tuning, including nested obstacles and approximate route restoration. | Camera/DepthNet and optional real base motion; no arm. | Reads only `obstacle_avoidance_parameters.json`; writes session log, frame CSV, and optional AVI under `logs/obstacle_avoidance_tuning/`. |

## Parameter and Reference Files

- `base_speed_time_parameters.json` defines standalone base calibration cases. Results are suggestions; they are not copied into production automatically.
- `obstacle_avoidance_parameters.json` isolates experimental obstacle geometry, confirmation, motion, HUD, and recording settings from the production configuration.
- `vague_map_test_parameters.json` defines the tester's initial pose, bin location, coarse navigation, and forward-factor measurement inputs.
- `useful_arm_states_record.json` is a historical pose reference containing manually recorded servo angles, command order, and pauses. It is independent of `empirical_parameters.json`, may include retired or duplicate state names, and must never be treated as the active arm configuration.

The active system configuration remains in the project root:

- `config.json` for paths, feature switches, dry-run policy, and mission policy;
- `empirical_parameters.json` for measured motion, vision, navigation, and arm values;
- `predefined_routines.json` for declarative search paths.

When a notebook says **Reload JSON**, edit the file first and reload it before starting a new session. Do not edit a JSON file while a motion worker is active.

## Helper Modules

The small notebooks intentionally delegate hardware lifecycle and calculations to testable Python helpers:

- `camera_interrupt_tools.py` — camera/DepthNet lifecycle UI.
- `diagnostic_tools.py` — SCSCL decoding, PnP statistics, candidate metrics, camera calibration, and CSV utilities.
- `motion_performance_tools.py` — base motion performance UI and interruptible motion jobs.
- `base_speed_time_calibration.py` — calibration case loading, execution, and CSV annotation; it also provides a command-line interface.
- `can_stable_stop.py` — live stop-threshold session, HUD, CSV, and recording lifecycle.
- `vague_map_tester.py` — isolated map-navigation configuration, preview, calibration, and run session.
- `bin_docking_tuning.py` — focused production docking runtime, compact status, disk logging, and media recording.
- `multi_detection_recording.py` — shared live/recording panels for dual-can and dual-obstacle tests.
- `obstacle_avoidance_tuning.py` — standalone obstacle analysis, phase control, nested bypass, restoration, HUD, and recording.

Normally, open the matching notebook instead of importing a helper manually. The exception is `base_speed_time_calibration.py`, whose safe command-line cases are documented in the project README.

## Logs and Recordings

Generated evidence is grouped by tool and timestamp under the project `logs/` directory. CSV rows and AVI frames from the same recording session are intended to be reviewed together. A video shows what the HUD reported; the CSV preserves the numeric values needed for analysis.

Always stop the session before copying or opening a recording. OpenCV does not finalize an AVI header until its writer is released, so an active recording may look missing or corrupt.

## Common Recovery Steps

- **Another stage is running:** press the notebook STOP action once and wait for the worker-finished message. Do not create a second panel in the same kernel.
- **Blank or stale live view:** release the current camera owner, then verify the camera in `camera_network_diagnostics.ipynb` before reopening the original tool.
- **`ValueError: I/O operation on closed file`:** a stale worker is still connected to an old notebook output stream. Stop hardware, restart that notebook's kernel, and create only one panel instance.
- **A button appears inactive:** check the notebook output for a rejected hardware flag, missing model/calibration, or an existing background job.
- **Recording does not open:** stop the run and wait for finalization, then use the notebook's recording link or open the path printed in the log.

If STOP does not promptly halt real base motion, use the physical power/control stop rather than waiting for the browser.
