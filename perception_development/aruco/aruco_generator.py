import os
import numpy as np
import cv2
from aruco_dict import ARUCO_DICT

aruco_type = "DICT_6X6_250"
tag_size = 250  # size of the marker in pixels has to match the size of the marker in the dictionary
id = 1

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

print("ARUCO marker of type '{}' with ID '{}'".format(aruco_type, id))
tag = cv2.aruco.generateImageMarker(arucoDict, id, tag_size)

# Save the tag to disk
os.makedirs("markers", exist_ok=True)
name = "markers/" + "aruco_{}_id_{}.png".format(aruco_type, id)


cv2.imwrite(name, tag)
cv2.imshow("ArUCo Tag", tag)
cv2.waitKey(0)
cv2.destroyAllWindows()