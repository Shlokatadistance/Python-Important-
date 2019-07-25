
import cv2 
import numpy as np
import imutils
  
FILE_NAME = 'reference.jpg'
try: 
    # Read image from disk. 
    img = cv2.imread(FILE_NAME)
    img = imutils.resize(img,height = 500)
  
    # Canny edge detection. 
    edges = cv2.Canny(img, 75, 200) 
  
    # Write image back to disk. 
    cv2.imshow('detected',edges)
except IOError: 
    print ('Error while reading files !!!') 
