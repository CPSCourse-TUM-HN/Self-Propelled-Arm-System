import cv2
import os

""" 
Collect images for camera calibration using a chessboard pattern
This script captures images from the webcam and saves them for calibration, but you can use this with any camera.
"""

# Directory to save images
SAVE_DIR = "calib_images"
# Number of images, use at least 15-20 images
NUM_IMAGES = 15   
# How many inner corners per chessboard row and column, not squares (IMPORTANT !!!)
CHESSBOARD_SIZE = (8, 6)   #meaning the chessboard is 9 x 7 squares

os.makedirs(SAVE_DIR, exist_ok=True)

# 0 = default webcam, if you have additional cameras, you can change this index
cap = cv2.VideoCapture(0)   
count = 0 # image counter

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #  Find chessboard corners with findChessboardCorners function
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    # If any corners are found, draw them 
    if found:
        cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, found)
        cv2.putText(display, "Press SPACE to save", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam", display)
    key = cv2.waitKey(1)

    # If SPACE is pressed and corners are found, save the image
    if key == 32 and found:  
        filename = os.path.join(SAVE_DIR, f"img_{count:02d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        count += 1
        # If image number reaches the limit, stop the process
        if count >= NUM_IMAGES:
            break

    # If ESC is pressed, exit the loop
    elif key == 27:  
        break

cap.release()
cv2.destroyAllWindows()
