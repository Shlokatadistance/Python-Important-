import imutils
import numpy as np
import cv2
image = cv2.imread('reference_1.jpg')
for angle in np.arange(0,360,15):
    rotated =  imutils.rotate_bound(image,angle)
    cv2.imshow("Rotated",rotated)
    cv2.waitKey(0)
