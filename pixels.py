"""
import numpy as np
import cv2

img = cv2.imread('reference12.jpg')
n = img.size
S=0;
B,G,R = cv2.split(img)
for i in range(1,n):

	S=S + format(img[100,50,0])[i] + format(img[100,50,1])[i] + format(img[100,50,2])[i]
	

Level = S/(3*n)
print(Level)

imahe = cv2.imread('download.jpg')
alpha = 1
beta = 0
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        for c in range(image.shape[2]):
            new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

import cv2
from matplotlib import pyplot as plt
import numpy as np

# Read image in BGR
img_path = 'download.jpg'
img = cv2.imread(img_path)

# Convert BGR to HSV and parse HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

# Plot result images
plt.imshow("Original", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.imshow("HSV", hsv_img)
plt.imshow("H", h)
plt.imshow("S", s)
plt.imshow("V", v)
plt.show()
"""
import cv2 
  
# importing library for plotting 
from matplotlib import pyplot as plt 
  
# reads an input image 
img = cv2.imread('download.jpg',0) 
  
# find frequency of pixels in range 0-255 
histr = cv2.calcHist([img],[0],None,[256],[0,256]) 
  
# show the plotting graph of an image 
plt.plot(histr) 
plt.show() 
