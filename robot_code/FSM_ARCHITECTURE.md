# Demo FSM Architecture

## Mission Model

`DemoStateMachine` is the only production coordinator. It owns a `MissionContext` containing the current target, a typed runtime `VagueMap`, grabbed state, completed pickup count, retries, state timing, and last observation.

The FSM definition and implementation are deliberately separated:

- `demo_core/fsm_types.py` defines `MissionState`, `MissionEvent`, `TargetType`, and `MissionContext`. It contains mission data and bookkeeping but no transition table.
- `demo_core/state_machine.py` defines `DemoStateMachine`, the transition table, state handlers, finalization sequence, retries, and cleanup.
- `demo_core/navigation.py` implements target search, alignment, approach, and final verification actions requested by the state machine.
- `demo_core/searching_routine.py` validates and incrementally executes low-priority base paths loaded from `predefined_routines.json`.
- `demo_core/vague_map.py` implements command odometry, approximate can memory/correction, and coarse point navigation.
- `demo_core/robot_control.py` translates those actions into base and servo commands.
- `demo_core/depth_vision.py` manages the JetBot camera and depthNet lifecycle. `demo_core/perception.py` converts camera/model results into common target observations.

The nominal transition chain is:

```text
IDLE -> INITIALIZING -> PLANNING
PLANNING -> SEARCHING | MAP_NAVIGATING | DONE
VERIFY_TARGET -> ALIGNING | SEARCHING
SEARCHING -> ALIGNING | PLANNING | INTERMEDIATE
SEARCHING -> AVOIDING -> SEARCHING for obstacle-aware predefined routines
MAP_NAVIGATING -> ALIGNING | SEARCHING | PLANNING | INTERMEDIATE
ALIGNING -> APPROACHING | SEARCHING | INTERMEDIATE
APPROACHING -> FINAL_VERIFY | SEARCHING | AVOIDING | INTERMEDIATE
FINAL_VERIFY -> FINALIZING | BIN_SIDE_DOCKING | SEARCHING | INTERMEDIATE
BIN_SIDE_DOCKING -> FINALIZING | INTERMEDIATE
AVOIDING -> VERIFY_TARGET | INTERMEDIATE
FINALIZING -> PLANNING
INTERMEDIATE -> PLANNING | FAILED
```

The first mission target is `CAN`. With no mapped candidate, `PLANNING` starts the selected search routine. A successful pickup sets `grabbed=true`, increments `completed_pickups`, and selects the bin docking pose for `MAP_NAVIGATING`. A successful release clears `grabbed`; the nearest mapped can is preferred, otherwise the search routine starts again.

Every map-navigation update is preceded by visual detection. A visible can or AprilTag immediately stops continuous map motion and transfers control to `ALIGNING -> APPROACHING -> FINAL_VERIFY`. Reaching only the approximate map destination enters `SEARCHING`; it never claims the target has been reached visually. Successful bin final verification normally resets odometry to `bin_docking_pose`. With exp2 enabled, the last stable side-view Tag observation may localize the base when `vague_map.bin_tag_localization.enabled=true`; otherwise reset is delayed until release and uses `bin_side_docking_pose`.

There is no score flag or score value. Detector confidence remains part of an observation and must not be interpreted as mission progress. The demo default is `max_pickups=1`, so one completed pickup and release ends the mission. With `max_pickups=0`, the map-candidate/search-routine cycle remains open-ended.

## Vague Map Data Flow

Base `pulse()` and continuous `start_motion()`/`stop()` report their effective motion to `CommandOdometry`, including in dry-run. Straight motion updates position using the calibrated meters-per-speed-second and slip factor; turns update heading and normalize it to `[-pi, pi)`.

Docking pose correction also rigidly transforms every remembered can coordinate using the old and corrected robot poses. This keeps cached candidates spatially consistent with the corrected robot frame while the surveyed bin pose remains fixed.

While carrying a can toward the bin, each camera frame is also passed to `CanDetector.detect_all()`. One DepthNet field is computed per frame, then sampled around each candidate center. Upper-image candidates are rejected to avoid mapping the carried can. Accepted world coordinates are bounds checked and merged by radius. This mapping is intentionally approximate and never replaces final visual navigation.

## Navigation Contract

Can and bin navigation use the same four-stage contract:

1. `SEARCHING`: detect first, then run one incremental step of the selected predefined base routine only while no target is accepted. Routine motion is the lowest-priority action and stops immediately on visual takeover or timeout.
2. `ALIGNING`: use normalized horizontal center error; turn continuously at `turn.slow` until inside tolerance, and stop immediately if visual tracking is lost.
3. `APPROACHING`: reacquire every frame, steer when necessary, and stop using the configured threshold. Can pickup supports `bbox_height`/DepthNet modes; bin docking uses calibrated AprilTag PnP `z` directly.
4. `FINAL_VERIFY`: repeat a tighter alignment for the required number of stable frames.

Can acceptance includes confidence thresholds. AprilTag acceptance is based on the configured tag ID. Both targets expose the same observation fields such as `found`, `bbox`, `center_x`, `error_x`, and normalized box height.

The navigator consumes a stable can observation contract and does not handle inference details. `CanDetector` is the sole adapter from native jetson-inference detections to that contract.

Search routines are declarative `drive(distance_m)` and `turn(angle_rad)` sequences. Linear completion uses the configured direction command scale and vague-map linear response; turn completion uses the measured direction/speed turn-response model. Routine 0 is the default unbounded left rotation with a large timeout. Routines 2 and 3 trace the requested counter-clockwise square from its left midpoint; routine 3 adds open-loop full turns at the lower and upper midpoints, so it is demonstrative rather than a precision heading test.

Routines 2 and 3 check depth obstacles before translation. An avoidance transition preserves the routine step and theoretical completed distance. On `ROUTINE_RESUME`, the avoidance odometry delta is appended to persistent routine memory. After the remaining sequence finishes, any displaced cycle performs coarse map navigation back to its saved origin and restores the origin heading. This prioritizes eventual return over retaining exact square geometry.

### Experimental Side Docking (exp2)

Exp2 preserves the normal front-facing bin search, alignment, approach, and final verification.
It then enters the explicit `BIN_SIDE_DOCKING` state:

```text
front final verify
  -> move Servo 1 to the fixed 90-degree side-view pose and wait for it to settle
  -> execute the fixed 90-degree direction-specific calibrated base turn
  -> wait for camera settling and discard configured stale frames
  -> staged base-only AprilTag yaw and center correction
  -> N frames with both yaw and center inside tolerance
  -> optional gated Tag-based map localization
  -> side_view_bin_insert
  -> open Servo 4 while retaining the insertion geometry
  -> safe_home
  -> preserve accepted visual pose, otherwise side-parking fallback reset
  -> DONE when max_pickups is reached
```

Side correction uses OpenCV PnP to estimate the signed left/right Tag-plane yaw. Continuous mode
commands one axis until vision accepts it, settles the camera, then checks the other axis; its
single-yaw-segment timeout is a safety/recheck bound, not an angle calculation. Pulse mode uses the
configured fixed yaw and center pulse durations in the same staged sequence. AprilTag PnP requires
the shared `camera.calibration_yaml`; missing or invalid calibration fails explicitly.
Servo 1 remains at `side_view_grabbing.s1` throughout side docking. The camera therefore has a
fixed transform relative to the base, raw camera yaw remains the base yaw error, and only the base
is allowed to correct it before release.
Longitudinal
correction uses normalized horizontal center error with the right-facing camera's reversed
forward/back mapping. Tag corner-angle and bbox-height metrics remain available for HUD diagnostics
but do not gate or command side docking.

The `side_view_grabbing` pose keeps the can held while turning the camera/arm to the configured
right-facing angle. The calibrated side-entry base turn cannot start until servo motion has settled. When
`vague_map.bin_tag_localization.enabled` is true, the final stable Tag pose is combined with the
surveyed marker pose and side-camera mount transform. A plausible result updates the map pose
before release and is preserved through the arm-only release sequence. Missing calibration,
incomplete observations, or rejected jumps retain the fixed fallback: reset to
`vague_map.bin_side_docking_pose` after release and `safe_home`. A failed side dock returns to
`carry` before retry handling; it does not issue an additional base recovery turn.

## Finalization

Can finalization is deterministic:

```text
safe_home (during initialization)
arm_down
wait for servo positions
optional base push
grab
carry
```

Bin finalization normally runs `release -> safe_home`. Exp2 runs
`side_view_bin_insert -> release(s4 only) -> safe_home`; it then preserves an accepted visual pose or applies the
fixed side-parking coordinate fallback. `carry` is used only by the exp2 failure recovery path.
Mechanical-arm tuning belongs in
`tuning_tools/arm_sequence_tuning.ipynb`; the FSM only consumes the resulting empirical parameters.

## Avoidance

Avoidance is configured by `avoidance.strategy`:

- `disabled`: approach never enters avoidance.
- `scripted`: fixed turn, forward, and rejoin pulses, then target reacquisition.
- `tangentbug_depth`: converts a depth profile into free-space gaps, chooses a local heading using target direction and clearance, and emits one incremental motion action. Optional overlays are written under `logs/`.

The planner is deliberately incremental. It does not claim metric localization or a complete global TangentBug implementation.

TangentBug and scripted avoidance remain independent planners and do not alter their paths to preserve
the routine geometry. When avoidance was entered from an obstacle-aware search routine, the routine
executor does use command odometry from the vague map to record the detour displacement and later
return to the saved cycle origin.

## Failure And Cleanup

Search/alignment/approach timeouts enter `INTERMEDIATE`. The failed target memory is cleared and the mission retries up to `runtime.retry_limit`; then it enters `FAILED`. Unexpected exceptions also enter `FAILED`.

`run()` always invokes `stop_all()` in `finally`, which stops the base, attempts `safe_home`, and releases the camera. Notebook users can also call `stop_all()` and `release_camera()` directly.

## Public API

```python
from demo_core import DemoDiagnostics, DemoStateMachine, load_config

config = load_config()
diagnostics = DemoDiagnostics(config)
state_machine = DemoStateMachine(config)

state_machine.step_once()
state_machine.run()
state_machine.stop_all()
state_machine.release_camera()
```

Configuration is merged as `empirical_parameters.json -> config.json -> overrides`. The old flat reference is not adapted or loaded.
