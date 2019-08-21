#detecting edges when more than one object is present
import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
image = cv2.imread('image.jpg')
ratio = image.shape[0]/500.0
image = imutils.resize(image,height = 500)
orig = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
retval,threshold = cv2.threshold(image,220,255,cv2.THRESH_BINARY_INV)
kernel = np.ones((5,5),np.unit8)
eroded = cv2.erode(image,kernel,iterations = 5)
#in the case i considered,i.e reference 7, 5 iterations wer necessary to effectively get the edges
#of the required areas, i am guessing that dilations may be of some help in the other cases too

greay = cv2.GaussianBlur(eroded,(5,5),0)
edged = cv2.Canny(grey,75,200)
cv2.imshow()
