# Focused nuXmv verification

This directory contains the smallest reproducible verification subset used in
the project report. It covers only three arguments:

1. Tag relocalization and the correctness of next-cycle map navigation;
2. the fixed can-pickup sequence under explicit physical contracts;
3. the bounded x/z/yaw pose reached by side docking.

The models abstract continuous distance, angle, image noise, and actuator
dynamics into finite predicates such as `x_ok`, `z_ok`, and
`robot_pose_bounded`. Therefore, the results verify controller logic under the
stated contracts; they do not replace calibration or physical experiments.

## Contents

```text
verification/
├── README.md
├── run_nuxmv.sh
└── models/
    ├── relocalization_navigation.smv
    ├── pickup_can.smv
    └── docking_pose_control.smv
```

`results/` is generated locally by the runner and is intentionally not part of
the source bundle.

## The three arguments

### 1. Relocalization and next-cycle navigation

Main reasoning chain:

```text
accepted Tag fix
-> Tag-derived robot pose
-> bounded robot/bin-relative pose under A_tag
-> correctness preserved to the next docking under A_nav
-> correct can-map use under P1_can
-> eventual next docking only under the separate A_progress contract
```

The model deliberately shows that changing the software pose source does not,
by itself, prove geometric accuracy. It also separates coordinate correctness
from progress: a correct map does not force the environment to produce the next
event.

### 2. Fixed can pickup

Main reasoning chain:

```text
visually stable can pose
-> arm down
-> fixed push
-> close gripper
-> carry pose
-> pickup complete with the can physically held
```

The ordering invariants are unconditional properties of the modeled state
machine. Physical pickup additionally requires `A_can_pose`,
`A_arm_geometry`, `A_push`, `A_grip`, and `A_actuator`. The expected
counterexample demonstrates that a software pickup marker is not physical
evidence that the can is held.

### 3. Physical side-docking pose

Main reasoning chain:

```text
PnP-z front stop establishes a longitudinal margin
-> calibrated 90-degree turn enters the side-view correction basin
-> closed-loop x/yaw correction reaches its tolerances
-> bounded x/yaw-to-z coupling preserves the longitudinal margin
-> DOCKED with x_ok, z_ok, and yaw_ok
```

The model reflects the controller asymmetry: z is established before side
docking, while x and yaw are directly corrected afterward. Without
`A_z_coupling`, x/yaw may be accepted while z has drifted outside its band;
that stronger statement is intentionally checked and expected to be false.

## Result interpretation

With nuXmv 2.1.0, the reference run produced:

| Model | TRUE | Expected FALSE | Unexpected FALSE | Total transition relation |
|---|---:|---:|---:|---|
| Relocalization/navigation | 10 | 4 | 0 | yes |
| Pickup | 7 | 2 | 0 | yes |
| Side docking | 7 | 4 | 0 | yes |

`TRUE` means the property holds on every execution admitted by that model and
its active assumptions. A property whose name begins with `EXPECTED_FALSE_`
intentionally removes one necessary contract. Its counterexample explains why
the stronger claim cannot be made. These counts are numbers of checked
properties, not experimental success rates.

## Install nuXmv and read the documentation

- Official download page: <https://nuxmv.fbk.eu/download.html>
- Official documentation index: <https://nuxmv.fbk.eu/documentation.html>
- Official user manual (PDF):
  <https://nusmv.fbk.eu/userman/v21/nusmv.pdf>

nuXmv is distributed under its own license; consult the official download page
and license before redistributing binaries. Keep the executable outside this
repository and pass its path to the runner.

## Run on Linux

Make the runner executable, then pass the nuXmv executable path:

```bash
chmod +x run_nuxmv.sh
./run_nuxmv.sh /path/to/nuXmv
```

If `nuXmv` is already on `PATH`, the argument may be omitted:

```bash
./run_nuxmv.sh
```

The script runs property checking with cone-of-influence reduction, checks that
the transition relation is total, stores raw output in `results/`, and fails if
the number of false properties differs from the explicitly named expected
counterexamples.

The equivalent commands for one model are:

```bash
/path/to/nuXmv -coi models/relocalization_navigation.smv
/path/to/nuXmv -ctt -is -ils -ii models/relocalization_navigation.smv
```

The raw `results/*.txt` files are reproducible build artifacts and are ignored
by Git by default. Commit them only when a course submission explicitly asks
for captured tool output; otherwise the models, runner, and summarized table
above are sufficient.

## Note on weak-until notation

For LTL propositions `p` and `q`, weak until is written here as:

```text
p W q  ==  (p U q) OR G(p)
```

`p U q` requires `q` eventually to occur and requires `p` before that point.
`p W q` is weaker: either `q` occurs with `p` holding beforehand, or `q` never
occurs and `p` remains true forever. The relocalization model uses the expanded
form rather than relying on a `W` parser token:

```smv
X ((bin_relative_correct U phase = DOCKED)
   | G bin_relative_correct)
```

The outer `X` matters: the guarantee starts in the state after the accepted Tag
fix. This property intentionally does not assert that another docking must
eventually occur; that liveness claim requires `A_progress` separately.

