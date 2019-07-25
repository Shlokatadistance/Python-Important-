import numpy as np 
import argparse
import cv2
#ap = argparse.ArgumentParser()
#ext parsing is a common programming task that splits the given sequence of characters or values (text) into smaller parts based on some rules. 
#ap.add_argument("-i","--image",required =True,
	#help="path to input image file")
#the basic idea is to put the text in the lighter font and the darken up th ebackgorund, thus realizing how
#far the text stretches in the plane. After the necessary observations, the textis rotated and fit into the plane 
#properly, thus limiting the usage of the reference 
#also remember THAT CV2 FILES MUST NEVER BE SAVED AS CV2.Py, they show the imread or other erros
#args = vars(ap.parse_args())

image = cv2.imread('sample.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#changing color spaces
gray = cv2.bitwise_not(gray)
#extracting out particular parts of the image, bitwise operators are use, bitwise_and,bitwise_or
thresh = cv2.threshold(gray,0,255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#based on the threshold values, the parts of the image are assigned black or white
coords = np.column_stack(np.where(thresh > 0))
#if i havve something like (1,2,3) and (4,5,6), it return 
#[1,2]
#[2,3]
#[4,5]
#stacking 1D arrays into 2D arrays, transforming one form into the other.
#But it doesnt really work that
angle = cv2.minAreaRect(coords)[-1]
#finding the minum area rectangle
if angle < -45:
	angle = -(90 + angle)
else:
	angle = -angle
(h,w) = image.shape[:2]
center = (w //2, h//2)
M = cv2.getRotationMatrix2D(center,angle,1.0)
#used for rotating image, with respect to center, angle is given and the 0 signified no scaling
rotated = cv2.warpAffine(image,M,(w,h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

cv2.putText(rotated,"Angle: {:.2f} degrees".format(angle), 
	(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
print("[INFO] angle: {:.3f}".format(angle))
cv2.imshow('original',image)
cv2.imshow("Rotated",rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
