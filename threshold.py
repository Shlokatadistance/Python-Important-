#this is the one for thresholding
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('watch.jpg')
grayscales  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#using imread grayscale is a better method in this case
retval,threshold = cv2.threshold(grayscales,12,255,cv2.THRESH_BINARY)
#the parameters in this are the image, the minimum thresh, maximum value, and any additional
#instruction from the developers side
cv2.imshow('original',img)
cv2.imshow('converted',threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
#when i convert using cv2.imread_grayscale, it gives a slightly darker output, with cv2.cvtColor
#the output still has some of its color
#make sure that the arguement in the threshold function is the grayscaled image
#and NOT the original image, otherwise you will end up getting the bad output
#the issue with grayscale was happening because i was not using the grayscale image in threshold

