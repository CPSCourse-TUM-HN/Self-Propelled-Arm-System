# Project Structure

```text
robot_code/
  config.json
  empirical_parameters.json
  predefined_routines.json
  run_demo.py
  demo_core/
    config.py
    diagnostics.py
    depth_vision.py
    fsm_types.py
    navigation.py
    searching_routine.py
    vague_map.py
    tangentbug.py
    state_machine.py
    tangentbug.py
    vague_map.py
    logging_utils.py
  assets/
    bin_apriltag_36h11_id_0.png
    models/detectnet_native_can/
    samples/can1.jpg
  tests/
    full_demo_integration_test.ipynb
    test_*.py
  scripts/
    board_first_checks.py
    home_pipeline_smoke.py
    validate_can_detectnet_image.py
  tuning_tools/
    camera_depthnet_interrupt_test.ipynb
    camera_interrupt_tools.py
    camera_network_diagnostics.ipynb
    base_motion_performance.ipynb
    motion_performance_tools.py
    base_speed_time_calibration.ipynb
    base_speed_time_calibration.py
    base_speed_time_parameters.json
    obstacle_avoidance_tuning.ipynb
    obstacle_avoidance_tuning.py
    obstacle_avoidance_parameters.json
    arm_sequence_tuning.ipynb
    bin_docking_tuning.ipynb
    bin_docking_tuning.py
  logs/
  legacy_codes.zip             optional historical archive; not a runtime dependency
```

## Ownership

- `run_demo.py` is the only complete demo entry point.
- `demo_core/state_machine.py` owns transitions, retries, pickup/release sequencing, and cleanup.
- `demo_core/fsm_types.py` defines mission states, events, target types, and shared mission context.
- `demo_core/navigation.py` implements shared can/bin navigation plus the optional exp2 side-docking controller.
- `demo_core/searching_routine.py` owns validation and incremental execution of externally defined low-priority search paths.
- `demo_core/vague_map.py` owns runtime map data, command odometry, mapped-can merging/correction, and coarse point navigation.
- `demo_core/diagnostics.py` owns reusable camera and model lifecycle operations for notebooks.
- `demo_core/perception.py` exposes can, AprilTag, and depth observations.
- `demo_core/robot_control.py` owns base pulses and servo commands.
- `demo_core/depth_vision.py` owns the JetBot camera and depthNet lifecycle and depth observations.
- `demo_core/tangentbug.py` is an optional depth-profile avoidance planner.

The old generic module names are no longer part of the public structure: FSM definitions live in `fsm_types.py`, FSM execution lives in `state_machine.py`, robot commands live in `robot_control.py`, and camera/depth handling lives in `depth_vision.py`. `perception.py` retains its standard robotics name.

Mission progress is represented by `MissionContext.completed_pickups` plus the runtime `VagueMap` cache state. Positive `runtime.max_pickups` is a completion cap; zero keeps the search/cache cycle open-ended. There is no separate score configuration.

`tests/` contains automated tests plus the full-demo integration acceptance Notebook. `scripts/` contains command-line preflight, smoke, and single-image validation entry points. `tuning_tools/` contains interactive camera, perception, base, arm, and navigation diagnostics that can access real hardware but are not production entry points. `arm_sequence_tuning.ipynb` is the single Notebook that deliberately modifies empirical arm pose parameters. None of these directories contains an alternative production FSM.

The can detector uses the standard jetson-inference `detectNet` interface and exposes the observation contract consumed by navigation. The runtime does not include a second can-inference backend.

## Generated And Historical Files

Runtime logs, diagnostic output, TensorRT engine files, notebook checkpoints, and Python bytecode are generated artifacts and must not be included in runtime synchronization commits.

Archived legacy code is not part of the runtime tree. If distributed with the project, it is kept in the optional
`legacy_codes.zip` archive rather than mixed with production modules. The archive's
`legacy_params/useful_arm_states_record.json` is a historical arm-tuning record: it helps relate old state names to
measured servo poses and command order, but production code must not load it or treat it as newer than
`empirical_parameters.json`. The large ZIP should be delivered through Git LFS or as a separate release artifact.
