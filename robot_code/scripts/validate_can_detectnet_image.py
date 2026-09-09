"""Run the configured Can DetectNet model against one image."""

from __future__ import print_function

import argparse
import json
import os
import sys

import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from demo_core import load_config
from demo_core.perception import CanDetector


def main():
    parser = argparse.ArgumentParser(description="Test the native detectNet can model on one image.")
    parser.add_argument("image")
    parser.add_argument("--output", default="detectnet_native_can_validation.jpg")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--clustering", type=float, default=None)
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print("[validate] could not read image: {}".format(args.image))
        return 1

    can_overrides = {}
    if args.threshold is not None:
        can_overrides["confidence_threshold"] = float(args.threshold)
        can_overrides["tracking_confidence_threshold"] = float(args.threshold)
    if args.clustering is not None:
        can_overrides["clustering_threshold"] = float(args.clustering)
    overrides = {
        "runtime": {"dry_run": {"camera": False, "base": True, "arm": True}},
        "detectors": {
            "can": dict({"enabled": True}, **can_overrides)
        },
    }
    result = CanDetector(load_config(overrides=overrides)).detect(frame)
    rendered = frame.copy()
    if result["found"]:
        left, top, right, bottom = [int(round(value)) for value in result["bbox"]]
        cv2.rectangle(rendered, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            rendered,
            "can {:.3f}".format(result["confidence"]),
            (max(0, left), max(20, top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not cv2.imwrite(args.output, rendered):
        print("[validate] could not write output: {}".format(args.output))
        return 1
    print("[validate] result={}".format(json.dumps(result, sort_keys=True)))
    print("[validate] output={}".format(os.path.abspath(args.output)))
    return 0 if result["found"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
