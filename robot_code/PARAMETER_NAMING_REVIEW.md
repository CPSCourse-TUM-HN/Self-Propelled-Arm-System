# Parameter and naming audit

This audit describes the active `vague_map_imp` configuration after the first cleanup pass. Runtime choices remain in `config.json`; measured/tuned values remain in `empirical_parameters.json`; search paths remain in `predefined_routines.json`.

## Active naming conventions

- Base commands use symbolic `linear.fast`, `linear.slow`, `turn.fast`, and `turn.slow` profiles. Direction-specific `base_motion_command_scales` are neutral empirical command multipliers, not friction coefficients.
- Target navigation is consistently divided into `search`, `align`, `approach`, and `near_align`. Both can and bin final confirmation read `near_align.required_stable_frames`; bin `near_align` inherits its center target and tolerances from `align`.
- Can stopping is described by `navigation.can.approach.stop.mode` plus explicit bbox/depth thresholds. Bin approach uses calibrated `pnp_z_threshold_m`.
- Side docking keeps both supported `correction_mode` values: `continuous` and `pulse`. Center and yaw parameters are named by the measured error they control.
- Arm poses have one operational release path: `side_view_bin_insert` keeps Servo 4 closed, then `release` opens Servo 4 at the same insertion geometry.

## Removed legacy concepts

The following are intentionally unsupported and raise `ValueError` if supplied by an override or an old merged JSON:

- fixed-time can `experimental_continuous` approach;
- AprilTag FOV-PnP fallback and manual yaw-zero workaround;
- docking height/standoff correction parameters;
- docking corner-angle completion gate and zero-valued approach `min_pulses` keys;
- final-yaw reliability gate;
- arm DepthNet grab verification and its target ROI/depth settings;
- `arm.gripper_speed`, `side_view_release_low`, and approach-level `final_verify_frames`.
- duplicate `detectors.bin.calibration_yaml`; all calibrated camera consumers use `camera.calibration_yaml`.

Unused methods `BaseController.drive_until()`, `BaseController.smooth_pulse()`, and `DepthSensor.sample_lens_center_for_hud()` were removed rather than retained as undocumented alternatives.

## Deliberately retained alternatives

- `continuous` and `pulse` side-docking correction are both active supported modes.
- `scripted` and `tangentbug_depth` obstacle avoidance remain selectable.
- AprilTag map localization remains configurable and requires calibration; it is enabled in the current demonstration parameters.
- Search routines 0, 2, 3, and 4 remain external declarative paths.
- Vague-map camera FOV remains in use for approximate can bearing; it is no longer used to synthesize camera intrinsics.

## Current high-impact values

- Bin front alignment offset: `-0.16` normalized x error.
- Bin PnP-z approach stop: `0.36 m` (latest remote hardware baseline).
- Side-docking x guard: `0.03` normalized error.
- Side-docking mode: `continuous`; pulse remains a supported fallback.
- Tag physical side length: `0.04 m`.

For operational values and their recommended Notebook, see `PARAMETER_QUICK_REFERENCE.md`.
