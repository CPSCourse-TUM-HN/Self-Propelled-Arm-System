from __future__ import print_function

import argparse
import json
import math
from pathlib import Path


ROUTINE_DRIVE_MULTIPLIERS = {
    2: (0.5, 1.0, 1.0, 1.0, 0.5),
    3: (0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5),
}


def set_square_lengths(data, side_length_cm):
    side_length_cm = float(side_length_cm)
    if not math.isfinite(side_length_cm) or side_length_cm <= 0.0:
        raise ValueError("side length must be a positive finite number of centimeters")

    side_length_m = side_length_cm / 100.0
    routines = {int(routine["id"]): routine for routine in data.get("routines", [])}
    for routine_id, multipliers in ROUTINE_DRIVE_MULTIPLIERS.items():
        if routine_id not in routines:
            raise ValueError("routine {} is missing".format(routine_id))
        routine = routines[routine_id]
        drive_steps = [step for step in routine["steps"] if step.get("motion") == "drive"]
        if len(drive_steps) != len(multipliers):
            raise ValueError(
                "routine {} has {} drive steps; expected {}".format(
                    routine_id, len(drive_steps), len(multipliers)
                )
            )
        for step, multiplier in zip(drive_steps, multipliers):
            step["distance_m"] = round(side_length_m * multiplier, 9)

    routines[2]["description"] = (
        "Start at the left-edge midpoint facing down, then trace a {:.6g} m "
        "counter-clockwise square at linear.fast speed.".format(side_length_m)
    )
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Set the square side length for predefined search routines 2 and 3."
    )
    parser.add_argument("length_cm", type=float, help="square side length in centimeters")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(__file__).resolve().with_name("predefined_routines.json"),
        help="routine JSON path (default: repository predefined_routines.json)",
    )
    args = parser.parse_args()

    path = args.file.resolve()
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    set_square_lengths(data, args.length_cm)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(
        "Updated routines 2 and 3 to a {:.6g} cm square in {}".format(
            float(args.length_cm), path
        )
    )


if __name__ == "__main__":
    main()
