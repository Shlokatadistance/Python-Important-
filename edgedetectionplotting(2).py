import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('sample.jpg')
gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
#in the above steps, we are converting the image to grayscale, and then to float32
corners = cv2.goodFeaturesToTrack(gray,4,0.1,100)
#goodfeaturestotrack is actually a thing btw
#parametes, img,max corners to detect,quality,minimum distance between corners
corners = np.int0(corners)
for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img,(x,y),3,255,-1)
    #we are actually putting a limit on the number of corners in the thing
    #the circle thing indicates that we iterate through each corner and then make a circle at each point
    #we think is a corner
cv2.imshow('corner',img)
           
#the thing actually points out the edges and corner and prints them out in the image,
#by highlighting htme with dots
