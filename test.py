import cv2
import numpy as np
def mapp(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmin(add)]

    diff = np.diff(h,axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmin(diff)]

    return hnew

image = cv2.imread('reference1.jpg')
img = cv2.imwrite('idk.jpg',img)
image = cv2.resize(image,(1300,800))
orig = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow('title',gray)

blurred = cv2.GaussianBlur(gray,(5,5),0)
cv2.imshow('blur',blurred)
edged = cv2.Canny(blurred,30,50)
#cv2.imshow('edges',edged)
contours,hierachy = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours,key = cv2.contourArea,reverse= True)
for c in contours:
    p = cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*p,True)

    if len(approx) == 4:
        target = approx
        break
approx = mapp(target)
pts = np.float32([[0,0],[4608,0],[4608,3456],[0,3456]])
op = cv2.getPerspectiveTransform(approx,pts)
dst = cv2.warpPerspective(orig,op,(10000,10000))
cv2.imshow('Scanned',dst)
