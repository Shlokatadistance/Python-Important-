import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('watch.jpg',0)
thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
cv2.imshow('adaptive',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
#adaptive threshold accounts for the curves in images, ensuring that areas do not blackout
#in the conversion process
#it utilised two additional functions , as shown in the code

