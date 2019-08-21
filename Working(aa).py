import cv2
import numpy as np
import imutils

def clahe(img, clip_limit=2.0, grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)

src = cv2.imread('reference12.jpg')
ratio = src.shape[0] / 500.0
orig = src.copy()
src = imutils.resize(src, height = 500)

# HSV thresholding to get rid of as much background as possible
hsv = cv2.cvtColor(src.copy(), cv2.COLOR_BGR2HSV)
lower_blue = np.array([0, 0, 120])
upper_blue = np.array([180, 38, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
result = cv2.bitwise_and(src, src, mask=mask)
b, g, r = cv2.split(result)
g = clahe(g, 5, (3, 3))

# Adaptive Thresholding to isolate the document
img_blur = cv2.blur(g, (9, 9))
img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 51, 2)

contours, hierarchy = cv2.findContours(img_th,
                                           cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)

# Filter the rectangle by choosing only the big ones
# and choose the brightest rectangle as the document
max_brightness = 0
canvas = src.copy()
for cnt in contours:
    rect = cv2.boundingRect(cnt)
    x, y, w, h = rect
    if w*h > 40000:
        mask = np.zeros(src.shape, np.uint8)
        mask[y:y+h, x:x+w] = src[y:y+h, x:x+w]
        brightness = np.sum(mask)
        if brightness > max_brightness:
            brightest_rectangle = rect
            max_brightness = brightness
        cv2.imshow("mask", mask)
        print(rect)
        #cv2.waitKey(0)
        break

x, y, w, h = brightest_rectangle
cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 1)
#cv2.imshow("canvas", canvas)
#cv2.imwrite(r'C:\Users\AdityaJoshi\Desktop\zapfin\mask.png',mask)
cv2.imshow('image',mask)
#cv2.imwrite(r'C:\Users\AdityaJoshi\Desktop\zapfin\mask.png',mask)
#cv2.waitKey(0)
