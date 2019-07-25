import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('watch.jpg',cv2.IMREAD_COLOR)
px = img[55,55]
img[55,55] = [255,255,255]
px = img[55,55]
print(px)
px = img[100:150,100:150]
print(px)
print(img.size)
print(img.shape)
#All of these are numpy arrays, that is why they are all stored in list style patterns
#Numpy arrays, printing them out, can give the specific parts of the image one is looking for
#gives a different output for px value
#similarly printing ou.t specific parts from the image is also possible
watch_face = img[37:111,87:194]
img[0:74,0:107] = watch_face
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#in my case the image is relatively lasrge, so only a specific part of the image
#is being highlighteD
retval,threshold = cv2.threshold(img,12,255,cv2.THRESH_BINARY)
cv2.
      


