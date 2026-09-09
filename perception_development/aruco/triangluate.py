import cv2
import numpy as np
import yaml

try:
    from plot_3d_points import (
        Aruco3DPlotter,
        build_claw_guidance,
        format_claw_guidance,
        format_marker_pose,
        to_camera_center_coordinates,
    )
except ImportError as exc:
    Aruco3DPlotter = None
    build_claw_guidance = None
    format_claw_guidance = None
    format_marker_pose = None
    to_camera_center_coordinates = None
    print(f"3D plotting disabled: {exc}")

# Load calibration data with Python's yaml
with open(r"calibration.yaml") as f:
    calib = yaml.safe_load(f)

""" 
load your camera’s intrinsic matrix and distortion coefficients from a YAML file
- camera_matrix is the 3×3 intrinsic parameters matrix.
- dist_coeffs holds lens distortion parameters.
- marker_length is the side length of the ArUco marker in meters.
"""
camera_matrix = np.array(calib["camera_matrix"])
dist_coeffs = np.array(calib["dist_coeff"])
marker_length = 0.020  # 20 mm
claw_id = 1

# Debug-only visualizer. Set this to False to disable the Matplotlib 3D plot
# when running with models or other real-time code that should not be slowed down.
ENABLE_3D_PLOT = True

# Converts OpenCV pose coordinates into camera-centered debug coordinates:
# center is (0, 0), left X is positive, right X is negative, down Y is positive,
# up Y is negative, and Z is positive going forward from the camera.
USE_CAMERA_CENTER_COORDINATES = True

"""
DICT_4X4_50 is a predefined dictionary of ArUco markers.
4x4 markers with 50 unique IDs.(16 bits per marker)
parameters are the default detection parameters.
"""
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# create a detector instance with default parameters.
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

cap = cv2.VideoCapture(0)
plotter = Aruco3DPlotter() if ENABLE_3D_PLOT and Aruco3DPlotter is not None else None

num=0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect markers in the frame with detectMarkers method
    corners, ids, _ = detector.detectMarkers(frame)
    """
    Example output of corners:
    (array([[[ 15., 339.],
        [ 91., 334.],
        [ 98., 404.],
        [ 19., 414.]]], dtype=float32),)
    """

    if ids is not None and len(ids) > 0:

        # Draw detected markers on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Define the 3D coordinates of the marker corners in the marker's coordinate system
        obj_points = np.array([
            [-marker_length / 2,  marker_length / 2, 0],
            [ marker_length / 2,  marker_length / 2, 0],
            [ marker_length / 2, -marker_length / 2, 0],
            [-marker_length / 2, -marker_length / 2, 0]
        ], dtype=np.float32)

        marker_positions = {}

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            image_points = marker_corners[0].astype(np.float32)

            """
            solvePnP estimates the pose of a 3D object given its 3D points and corresponding 2D image points.
            It returns the rotation vector (rvec) and translation vector (tvec).
            """
            retval, rvec, tvec = cv2.solvePnP(obj_points, image_points, camera_matrix, dist_coeffs)
            
            if retval:
                # Draw the axis on the frame
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                
                # Extract the translation vector and calculate the distance
                x, y, z = tvec.flatten()
                marker_position = np.array([x, y, z])

                if USE_CAMERA_CENTER_COORDINATES and to_camera_center_coordinates is not None:
                    marker_position = to_camera_center_coordinates(marker_position)

                marker_positions[marker_id] = marker_position
               
                # Print to terminal
                if format_marker_pose is not None:
                    print(format_marker_pose(marker_id, marker_positions[marker_id]))
               
                # Display on frame with different colors
                # org = (20, 40)  
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # font_scale = 0.7
                # thickness = 2
                # cv2.putText(frame, f"X = {x:.3f} m", (org[0], org[1]), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                # cv2.putText(frame, f"Y = {y:.3f} m", (org[0], org[1]+30), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
                # cv2.putText(frame, f"Depth = {z:.3f} m", (org[0], org[1]+60), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
                # cv2.putText(frame, f"Dist = {distance:.3f} m", (org[0], org[1]+90), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

        if claw_id in marker_positions and build_claw_guidance is not None:
            claw_guidance = build_claw_guidance(marker_positions, claw_id)

            for line in format_claw_guidance(claw_guidance):
                print(line)

        if plotter is not None:
            try:
                plotter.update(marker_positions, claw_id)
            except ImportError as exc:
                print(f"3D plotting disabled: {exc}")
                plotter = None

    # Display the frame with detected markers and pose estimatio
    cv2.imshow('ArUco Pose Estimation', frame)
     
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
if plotter is not None:
    plotter.close()
cv2.destroyAllWindows()
