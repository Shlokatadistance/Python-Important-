import cv2
import numpy as np
from matplotlib import pyplot as plt
#matplotlib is basically used to display frames in images and videos
#imread color is color without any alpha channel, which is the degree of opaqueness
#if you want to retain the opaqueness, use imread_unchanged
img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)
#so imread grayscale is converting the image to grayscale
#you can also use numbers, 1 - color, 0 -  grayscale and -1 = imread_unchanged
#so the code is img = cv2.imread('image.jog',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
