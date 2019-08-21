import numpy as np
import cv2
import imutils
image = cv2.imread('reference_11.jpg')
image = imutils.resize(image,height = 500)
"""
image[np.where((image == [255,255,255]).all(axis = 2))] = [10,10,10]
cv2.imshow('output', image)

lower_black = np.array([0,0,0], dtype = "uint16")
upper_black = np.array([70,70,70], dtype = "uint16")
black_mask = cv2.inRange(image, lower_black, upper_black)
cv2.imshow('mask0',black_mask)
"""

 
class ShapeDetector:
    
    
    def __init__(self):
        pass
         
        
     def detect(self, c):
         
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
 
		# if the shape has 4 vertices, it is either a square or
		# a rectangle
        elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
	ar = w / float(h)
	shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
	shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
 
			
			 
		# if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        return shape
    


            
