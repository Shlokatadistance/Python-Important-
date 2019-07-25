import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('watch.jpg',0)
#now drawing a random rectangle in the image using cv2
cv2.rectangle(img,(15,25),(200,150),(0,0,255),15)
#tha parameters here are the img, top left, bottom right, color and line thickness
#any shape can be drawn in cv2 using cv2.shape(), the only things you need to take care of are
#the parameters
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
