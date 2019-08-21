import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
img = cv2.imread('reference3.jpg')
gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#gray = np.float32(gray)
kernel = np.ones((6,6),np.uint8)
eroded = cv2.erode(gray,kernel,iterations = 11)

gray2 = cv2.Canny(eroded,75,200)
#eroded = cv2.erode(gray2,kernel,iterations = 20)
gray3 = imutils.resize(gray2,height = 600)
cv2.imshow('outlines',gray3)
#in the above steps, we are converting the image to grayscale, and then to float32
corners = cv2.goodFeaturesToTrack(gray,10000,0.01,10)
#goodfeaturestotrack is actually a thing btw
#parametes, img,max corners to detect,quality,minimum distance between corners
corners = np.int0(corners)
for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img,(x,y),10,255,-1)
    #we are actually putting a limit on the number of corners in the thing
    #the circle thing indicates that we iterate through each corner and then make a circle at each point
    #we think is a corner
#c = imutils.resize(img,height = 500)
#cv2.imshow('corner',c)
           
#the thing actually points out the edges and corner and prints them out in the image,
#by highlighting htme with dots
