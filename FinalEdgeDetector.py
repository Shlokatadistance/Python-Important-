#this is a cleaner version of the probablyworking.py code

import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
image = cv2.imread('reference5.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
orig = image.copy()
image = imutils.resize(image,height = 500)
retval,threshold = cv2.threshold(image,75,255,cv2.THRESH_BINARY)
grey = cv2.GaussianBlur(image,(5,5),0)
edges = cv2.Canny(grey,75,225)
cv2.imshow('edged image',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
#part 2 detecting the contours and and accordingly mapping into the rectangle
#this program doesnt use the minarea rect function to create the map
cnts = cv2.findContours(edges.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=  imutils.grab_contours(cnts)
cnts = sorted(cnts,key = cv2.contourArea,reverse = True)[:5]
for c in cnts:
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,0.02*peri,True)
    if len(approx) == 4:
        screenCnt = approx
        break
    else:
        print("No four required edges have been correctly found")


#Showing the contours and effectively generating the outline
cv2.drawContours(image,[screenCnt],-1,(0,255,0),2)
cv2.imshow('outlinedimage',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#next part is including that erosion part, in case where more than 2 images or edges are being detecte
#of two different objects
kernel = np.ones((5,5),np.unit8)
eroded = cv2.erode(image,kernel,iterations = 5)
edges = cv2.Canny(eroded,75,200)
cv2.imshow('edged for specific cases',edges)


