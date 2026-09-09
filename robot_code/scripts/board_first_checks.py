"""Report whether board-only camera, inference, base, and arm dependencies are available."""

import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    print("[check] Python board dependency check")

    try:
        import cv2
        print("[check] cv2 {}".format(cv2.__version__))
        print("[check] has aruco: {}".format(hasattr(cv2, "aruco")))
        if hasattr(cv2, "aruco"):
            print("[check] has ArucoDetector: {}".format(hasattr(cv2.aruco, "ArucoDetector")))
            print("[check] has AprilTag36h11: {}".format(hasattr(cv2.aruco, "DICT_APRILTAG_36h11")))
            if hasattr(cv2.aruco, "DetectorParameters"):
                print("[check] has DetectorParameters: True")
            else:
                print("[check] has DetectorParameters_create: {}".format(hasattr(cv2.aruco, "DetectorParameters_create")))
    except Exception as exc:
        print("[check] cv2 failed: {}".format(exc))

    try:
        from pupil_apriltags import Detector
        print("[check] pupil_apriltags import ok")
        detector = Detector(families="tag36h11", nthreads=1)
        print("[check] pupil_apriltags tag36h11 detector ok")
        del detector
    except Exception as exc:
        print("[check] pupil_apriltags failed: {}".format(exc))

    try:
        from jetbot import Camera, Robot
        print("[check] jetbot Camera/Robot import ok")
    except Exception as exc:
        print("[check] jetbot import failed: {}".format(exc))

    try:
        from SCSCtrl import TTLServo
        print("[check] SCSCtrl TTLServo import ok")
    except Exception as exc:
        print("[check] SCSCtrl import failed: {}".format(exc))

    try:
        from jetson_inference import depthNet, detectNet
        print("[check] jetson_inference depthNet/detectNet import ok")
    except Exception as exc:
        print("[check] jetson_inference import failed: {}".format(exc))

    try:
        from jetson_utils import cudaFromNumpy
        print("[check] jetson_utils cudaFromNumpy import ok")
    except Exception as exc:
        print("[check] jetson_utils import failed: {}".format(exc))

    for relative_path in [
        "assets/bin_apriltag_36h11_id_0.png",
        "assets/models/detectnet_native_can/can_ssd_mobilenet_v1.onnx",
        "assets/models/detectnet_native_can/labels.txt",
        "config.json",
        "empirical_parameters.json",
    ]:
        path = os.path.join(PROJECT_ROOT, relative_path)
        print("[check] exists {}: {}".format(relative_path, os.path.exists(path)))


if __name__ == "__main__":
    main()
