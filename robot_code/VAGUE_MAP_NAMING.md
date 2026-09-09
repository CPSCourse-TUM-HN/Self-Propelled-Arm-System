# Vague Map Naming

This document fixes the names used by the runtime map implementation. The map is deliberately approximate: it is command-based spatial memory for coarse travel, not SLAM, localization, or obstacle planning.

## Core Types

- `VagueMap`: runtime-only map state. It owns the robot pose, bounds, bin locations, and mapped cans. Alternatives considered: `ApproximateMap`, `SpatialMemory`.
- `Point2D`: a position containing `x_m` and `y_m`, with no heading. Alternatives: `MapPoint`, `Coordinate2D`.
- `Pose2D`: a robot or docking pose containing `x_m`, `y_m`, and `heading_rad`. Alternatives: `RobotPose`, `PlanarPose`.
- `MappedCan`: one approximate can entry with a unique ID, averaged position, confidence, observation count, and last-seen time. Alternatives: `CanLandmark`, `CanMemory`.
- `CommandOdometry`: updates `Pose2D` from issued base commands and calibrated motion coefficients. Alternatives: `OpenLoopOdometry`, `BaseCommandTracker`.
- `VagueMapNavigator`: rotates and advances toward a `Point2D`. It does not perform visual alignment, arm control, or avoidance. Alternatives: `CoarseMapNavigator`, `WaypointNavigator`.

## Mission States

- `SEARCHING`: executes the selected predefined routine whenever no mapped can is available or a map destination did not reveal its target.
- `MAP_NAVIGATING`: travels coarsely toward a mapped can or the bin docking position. Vision is checked before every movement update and has priority.

`SEARCHING`, `ALIGNING`, `APPROACHING`, and `FINAL_VERIFY` retain their existing visual meanings. Reaching a map coordinate enters visual search; it does not imply visual success.

Obstacle-aware search routines retain their origin pose, interrupted step progress, and theoretical avoidance displacement in `MissionContext.searching_routine_data`. This memory survives only the `SEARCHING -> AVOIDING -> SEARCHING` detour and is reset for a genuinely new search.

## Bin Coordinates

- `bin_marker_position`: world pose of the physical AprilTag marker. The retained name is historical; it now contains `heading_rad`, defined as the world heading of the marker-frame `+Z` axis used by OpenCV PnP.
- `bin_docking_pose`: desired robot position and heading for release. It is the coarse map-navigation destination and the fallback pose written back after successful front visual docking.
- `bin_side_docking_pose`: fallback robot pose after successful side docking. When calibrated Tag localization is enabled and passes plausibility gates, the measured robot pose is preserved instead.
- `bin_tag_localization.camera_mounts`: fixed camera optical-center offset and optical `+Z` heading in the robot base frame, separately calibrated for front and side arm poses.

The two values must not be treated as synonyms: the tag is on the bin, while the docking pose is where the robot should stop.

Tag localization is deliberately optional. It requires camera intrinsics, the known Tag world pose, and the camera-to-base mount for the active arm pose. Missing or implausible measurements fall back to the configured docking pose.

Every accepted docking pose correction applies the same planar rigid transform to all `MappedCan.position` entries. Their robot-relative direction and distance at correction time are therefore preserved; fixed bin coordinates remain unchanged.

## Coordinate Convention

The map is a fixed world frame initialized from the robot's starting frame; its axes do not rotate as the robot moves. The manually placed initial robot pose is `(0, 0, 0)`. `+Y` points along the initial camera/robot forward direction, `+X` points right, and positive heading rotates clockwise from `+Y` toward `+X`. Positions use meters and headings use radians.

The same axis convention is used for camera mounts and robot-relative examples. Thus, after a right-side docking, a bin directly to the robot's right has relative coordinates `(a, 0)` where `a > 0`.

Image horizontal error uses the existing convention: positive `error_x` means the target is right of image center. Mapping converts it with:

```text
bearing_offset = error_x * camera_horizontal_fov_rad
```

Therefore an image-right target produces a positive heading offset when the robot initially faces `+Y`.
