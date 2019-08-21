import cv2
import imutils
import numpy as np
import os

 
def transform(pos):
# This function is used to find the corners of the object and the dimensions of the object
    pts=[]
    n=len(pos)
    for i in range(n):
        pts.append(list(pos[i][0]))
       
    sums={} 
    diffs={}
    tl=tr=bl=br=0

    for i in pts:
        x=i[0]
        y=i[1]
        sum=x+y
        diff=y-x
        sums[sum]=i
        diffs[diff]=i
    sums=sorted(sums.items())
    diffs=sorted(diffs.items())
    n=len(sums)
    rect=[sums[0][1],diffs[0][1],diffs[n-1][1],sums[n-1][1]]
    #      top-left   top-right   bottom-left   bottom-right
#distance formula between the various sides
    h1=np.sqrt((rect[0][0]-rect[2][0])**2 + (rect[0][1]-rect[2][1])**2)     #height of left side
    h2=np.sqrt((rect[1][0]-rect[3][0])**2 + (rect[1][1]-rect[3][1])**2)     #height of right side
    h=max(h1,h2)
   #the figure is a rectangle
    w1=np.sqrt((rect[0][0]-rect[1][0])**2 + (rect[0][1]-rect[1][1])**2)     #width of upper side
    w2=np.sqrt((rect[2][0]-rect[3][0])**2 + (rect[2][1]-rect[3][1])**2)     #width of lower side
    w=max(w1,w2)
   
    return int(w),int(h),rect
"""
def erosionneeded(image):
    kernel = np.ones((5,5),np.uint8)
    eroded_image = cv2.erode(image,kernel,iterations = 6)
    return eroded_image
    """
def rotation(image,angle):
    h,w = image.shape[:2]
    center = (w/2,h/2)
    #angle90 = 90
    #angle270 = 270
    #angle180 = 180
    scale = 1.0
    rotated_image = imutils.rotate_bound(image,angle)
    #M = cv2.getRotationMatrix2D(center,angle,scale)
    #rotated_image = cv2.warpAffine(image,M,(w,h))
    #cv2.imshow('rotation',rotated_image)
    return rotated_image
   #the rotation is anticlockwise
#you can also use the negative angles to get the answers 
#using the imutils way is clockwise
#warpAffine mthod cuts out part of the image if it goes out of frame
img=cv2.imread('reference12.jpg')
r=500.0 / img.shape[1]
dim=(500, int(img.shape[0] * r))
img=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
kernel = np.ones((5,5),np.uint8)

 
cv2.imshow('INPUT',img)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(11,11),0)
#eroded = cv2.erode(gray,kernel,iterations = 6)
edge=cv2.Canny(gray,50,200)
cv2.imshow('edges',edge)
_,contours,_=cv2.findContours(edge.copy(),1,1)
cv2.drawContours(img,contours,-1,[0,0,255],2)
cv2.imshow('Contours',img)

n=len(contours)
max_area=0
pos=0
for i in contours:
    
    area=cv2.contourArea(i)
    if area>max_area:
        
        max_area=area
        pos=i
peri=cv2.arcLength(pos,True)
approx=cv2.approxPolyDP(pos,0.02*peri,True)
 
size=img.shape
w,h,arr=transform(approx)
 
pts2=np.float32([[0,0],[w,0],[0,h],[w,h]])
pts1=np.float32(arr)
M=cv2.getPerspectiveTransform(pts1,pts2)
dst=cv2.warpPerspective(img,M,(w,h))
image=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
#image=cv2.adaptiveThreshold(image,255,1,0,11,2)
#image = cv2.threshold(image,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
image = cv2.resize(image,(w,h),interpolation = cv2.INTER_AREA)
image = imutils.resize(image,height = 500)
cv2.imshow('OUTPUT',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

a = input("do you need to rotate the image?")
if a == "Yes":
    ang = int(input("Enter the rotation angle:"))
    image_new = rotation(image,ang)
    cv2.imshow('final output',image_new)
else:
    print("You didnt say anything else")



