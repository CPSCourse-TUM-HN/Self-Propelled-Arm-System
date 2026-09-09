import numpy as np
import cv2
import glob
import yaml

"""
After capturing images of a chessboard pattern, this script calibrates the camera
and saves the calibration parameters to a YAML file.
"""

# Chessboard settings
CHESSBOARD_SIZE = (8, 6) #meaning the chessboard is 9 x 7 squares
SQUARE_SIZE = 17  # mm (squire size of the chessboard squares, adjust as needed)

# Prepare object points based on the chessboard size and square size
# objp will hold the 3D coordinates of the chessboard corners in the world space
objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)

# Generate the grid points in the chessboard pattern
# objp[:, :2] will hold the x, y coordinates of the corners
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

# Scale the points by the size of each square
objp *= SQUARE_SIZE

# Arrays to store points
objpoints = []  # 3D points
imgpoints = []  # 2D points

# Load images from the specified directory
images = glob.glob("calib_images/*.jpg")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    # If corners are found, continue
    if found:
        objpoints.append(objp)
        # Refine the corner locations
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        # Append the refined corners to imgpoints list
        imgpoints.append(corners2)

# Calibrate the camera using the collected object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# print the calibration results
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)
print("Reprojection error:", ret)

calib_data = {
    "camera_matrix": mtx.tolist(),
    "dist_coeff": dist.tolist(),
    "reprojection_error": float(ret)
}

# Save the calibration data to a YAML file
with open("calibration.yaml", "w") as f:
    yaml.dump(calib_data, f)

print("Saved calibration.yaml")
