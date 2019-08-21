import numpy as np
import imutils
import cv2
from skimage.filters import threshold_local
image = cv2.imread('reference.jpg')
ratio = image.shape[0] / 500.0
image = imutils.resize(image,height = 500)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
retval, threshold = cv2.threshold(gray,75,225,cv2.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8)
#erosion = cv2.erode(gray,kernel,iterations = 1)
#erosion kind of destroys the image
#dilation = cv2.dilate(gray,kernel,iterations = 1)
#dilation removes some of the external news
edges = cv2.Canny(gray,75,200)
cnts = cv2.findContours(edges.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for c in cnts:
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,0.02 *peri,True)
    x,y,w,h = cv2.boundingRect(approx)
    if h>=5:
        rect = (x,y,w,h)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
cnts = imutils.grab_contours(cnts)

cv2.imshow('input',image)
#cv2.imshow('eroded',erosion)
#cv2.imshow('dilated',dilation)
#cv2.imshow('edges',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
pixeldata = image.data()
count = 0
for x,y in pixeldata:
    if pixel[x,y] == [255,255]
    print("The given pixel is white")
    count++
if count >20:
    print("The background is white")
