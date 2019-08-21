import cv2
import numpy as np
image = cv2.imread('outline1.jpg')
mask = np.zeros(image.shape[:2],np.uint8)
bgModel = np.zeros((1,65),np.float64)
fgModel = np.zeros((1,65),np.float64)
rect = (161,79,150,150)
cv2.grabCut(image,mask,rect,bgModel,fgModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0),0,1).astype('uint8')
image = image*mask2[:,:,np.newaxis]
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
