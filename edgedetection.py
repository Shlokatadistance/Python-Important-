
import cv2 
import numpy as np 
  
FILE_NAME = 'volleyball.jpg'
try: 
    # Read image from disk. 
    img = cv2.imread(FILE_NAME) 
  
    # Canny edge detection. 
    edges = cv2.Canny(img, 100, 200) 
  
    # Write image back to disk. 
    cv2.imwrite('result.jpg', edges) 
except IOError: 
    print ('Error while reading files !!!') 