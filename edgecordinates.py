import numpy as np
import cv2
import imutils
list = []
image = cv2.imread('reference5.jpg',0)
edges = cv2.Canny(image,100,255)

#indices = np.where(edges != [0])
#coordinate = zip(indices[0],indices[1])
#print(coordinate)
