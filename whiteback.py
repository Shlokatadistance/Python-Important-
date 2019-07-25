#this is the detection for lighter backgrounds
import numpy as np
import cv2
from skimage.filters import threshold_local
import imutils
import glob
imge = cv2.imread('reference4.jpg',0)
ratio = imge.shape[0]/500.0
imge= imutils.resize(imge,height = 500)
threshold = cv.threshold(image,75,150,cv2.THRESH_BINARY_INV)
#equ = cv2.equalizeHist(image)
#res = np.hstack((image,equ))

"""
th3 = cv2.adaptiveThreshold(image,300,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
edged = cv2.Canny(image,75,200)
cv2.imshow('edged',edged)
cv2.imshow('original',image)
#cv2.waitKey()
#cv2.destroyAllWindows()
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
"""
#check what kind of background you are getting
#use threshing
#edges = cv2.Canny(image,75,200)
def auto_canny(image,sigma = 0.33):
    v = np.median(image)
    lower = int(max(0,(1.0 - sigma)*v))
    higher = int(min(255,(1.0 + sigma)*v))
    edged = cv2.Canny(image,lower,higher)
    return edged
def whitebackground(detection):
    lower = int(max(0,max(two iterations)
#checking white background, using thresh inversion ,

edges = cv2.Canny(imge,10,225)
cv2.imshow('edged',imge)
cv2.waitKey()
cv2.destroyAllWindows()


