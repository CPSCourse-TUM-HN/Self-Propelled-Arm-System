# Full Demo Integration Test

For a short desk-side check before running this Notebook, use:

```bash
python3 scripts/home_pipeline_smoke.py
```

It keeps the base and arm in dry-run, probes each real camera/model path only once, and then validates the complete state chain with simulated hardware.

`tests/full_demo_integration_test.ipynb` is a thin board-side wrapper around production `DemoStateMachine`. It deliberately has no parameter widgets, preventing UI labels from drifting away from JSON keys.

## Iteration Loop

1. Edit and save `config.json` or `empirical_parameters.json`.
2. Click Reload Configuration in the notebook.
3. Run preflight or one FSM step.

With vague-map mode enabled, also inspect `[map]` lines for pose updates, mapped-can additions/merges/corrections, map target selection, and the bin docking pose reset. Map movement is only coarse positioning; the expected handoff is `MAP_NAVIGATING -> ALIGNING` as soon as vision sees the target.
4. Run the full demo when camera/model checks pass.
5. Inspect the notebook output and the timestamped file in `logs/`.

Reloading creates a fresh runtime from disk. Detector objects are reused during a runtime unless the reload-model option is selected. The notebook never writes configuration.

`Run Full Demo` runs in a background thread so the Notebook kernel remains available
to process safety-button callbacks. `STOP BASE` immediately stops the tracks and
pauses the state machine; use `Resume` to continue. `STOP ALL` requests termination
and runs the normal base, arm, and camera cleanup.

`STOP ARM` pauses the state machine and commands servos 1-5 to hold their currently
reported raw positions. Resuming retries an interrupted named pose from its beginning.
Because pausing is mission-wide, `STOP ARM` also stops the base.

Enable `live_stream` and/or `record_camera` before starting a run. The media monitor
starts automatically with the full demo, or can be controlled with `Start Media` and
`Stop Media`. Recordings use MJPEG AVI and are written under `diagnostic_outputs/`.
`media_fps` controls the 640x480 recording frame rate. `preview_fps` independently
controls the Notebook preview, which is downsampled to 320x240 and JPEG quality 55.
The default 1 FPS preview is intentionally a low-cost sampled monitor. After stopping
the recording, click `Recording Link` to display a downloadable link in the log output.

For stationary camera/model checks, leave `static_detection` enabled, choose Can,
Tag, or both in `static_target`, and click `Start Media`. The production detectors
are loaded and their labeled boxes appear in the same preview and recording HUD;
`detection_fps` controls inference sampling independently from preview/recording FPS.
This read-only mode does not update mission target memory or command the robot.

The `hud` switch renders the original clear 640x480 overlay for the recording. The
Notebook preview displays a downsampled copy of that same HUD.
It shows the current FSM state, target, cached detection bbox/center/confidence,
camera-center ROI, center depth, pickup state, and recording indicator. Center depth
is read only from the latest FSM cache. While the mission worker is active, the media
thread only displays the FSM's cached observations and never invokes a detector. Static
inference is allowed only while the mission worker is idle, preventing concurrent calls
to Jetson inference objects. A stale-frame warning is logged if the camera image remains
byte-identical for two seconds.

Normal can/bin approach now keeps a constant forward command active while each new
frame checks target visibility, steering error, stop threshold, and timeout. A
steering correction stops the straight command, executes the slow in-place turn
pulse, and then resumes forward motion. Fast search turns are also continuous until
the target or search timeout is reached; slow precision turns remain pulsed.

The `exp2` switch enables experimental right-facing bin docking after the normal front
approach and near alignment. It moves to `arm.poses.side_view_grabbing`, waits for the servos
to settle, and only then runs the configured supportive base turn. Base-only calibration uses
AprilTag center error, OpenCV PnP left/right yaw error, corner-angle error, and tag height.
Multiple stable frames are required before the arm enters `side_view_bin_insert`, waits for that
pose to settle, opens only Servo 4, and finally returns to `safe_home`.
The completed side-parking pose becomes the vague-map coordinate reset point; the successful
path does not apply another heading-reset turn.

Wait for the `BACKGROUND RUN FINISHED` message before resetting the runtime or
starting another full run. A runtime that has reached `DONE` or `FAILED` must be
reset and reloaded before it can run again.

## Safety

Dry-run flags come from `config.json`. Start with all three enabled. Enable the camera, base, and arm separately as confidence grows. The arm should remain dry-run until `tuning_tools/arm_sequence_tuning.ipynb` has confirmed `safe_home` and the pickup sequence.

Every complete run executes cleanup. For interrupted notebook work, use Stop All and Release Camera before restarting a kernel. `STOP BASE` is a pausing control rather than mission termination. If the Python process is terminated outside its cleanup path, release the camera before opening another notebook.

## Expected Results

- A dry-run reaches `DONE` with one simulated pickup and release.
- A real preflight reports a camera frame plus can, tag, and optional depth observations.
- A failed real mission reports the final state and reason, stops the base, attempts `safe_home`, and closes the camera.
- Logs use `[fsm]`, `[mission]`, `[approach]`, `[can]`, `[tag]`, `[depth]`, `[arm]`, `[base]`, and `[tangentbug]` prefixes. `[mission] completed_pickups=N` reports mission progress after a successful pickup.
