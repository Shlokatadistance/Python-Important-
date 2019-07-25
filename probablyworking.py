import cv2
import imutils
import numpy as np
from skimage.filters import threshold_local
#from pyimagesearch.transform import four_point_transform
image = cv2.imread('reference.jpg')
ratio = image.shape[0]/500.0
#the ratio thing is perfectly alright

orig = image.copy()
image= imutils.resize(image,height = 500)
#it basically just determines the size at which the image is displayed

#converting into grayscale and finding the edges
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
retval,threshold = cv2.threshold(gray,75,255,cv2.THRESH_BINARY)
#also here, the thresh values need to be refreshed
gray = cv2.GaussianBlur(gray,(5,5),0)
#even this one, the gray isnt working properly
#the gaussian blur is creating issues while detecting the edges
#the purpose of the blur is to remove the noise in the picture
#two techniques can be used, erosion and dilation

edged = cv2.Canny(image,75,200)
#edge detected image
cv2.imshow('image',image)
cv2.imshow('edges',edged)
#cv2.imwrite('edges1.jpg',edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

#step 2 , finding the contours in the picture
#we assueme that the largest triangle in the contour is the paper with its four edges
#finding contours in the edged image
cnts = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#retr list or retr external
#also here, the modification needs to be done
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts,key = cv2.contourArea,reverse = True)[:5]
#changing countour area to contour length

#now looping over the contours
for c in cnts:
    peri = cv2.arcLength(c,True)
   #using epsilon method
    approx = cv2.approxPolyDP(c,0.02 *peri,True)
    #swapping between true and false,not working
    #there needs to be made a modification here

    if len(approx) == 4:
        screenCnt = approx
        break
    else:
        print("no four vertices have been properly found")
#show the contour of the paper
#cv2.boxpoint(cnts)
cv2.drawContours(image,[screenCnt],-1,(0,255,0),2)
#cv2.imwrite('outline1.jpg',image)
cv2.imshow('outline',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
