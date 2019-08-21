import cv2
import imutils
import numpy as np
from PIL import Image
from PIL import ImageEnhance

image = cv2.imread("./paper12.jpg")
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#---BLUR--
def blur(image):
    threshold = 1500
    score = cv2.Laplacian(image, cv2.CV_64F).var()
    if score < threshold:
        print ("Not Blur")
        print("Blur score:",score)
        return 0
    else:
        print ("Blur")
        print("Blur score:",score)
        return 1

def coordinates(image):
    im=image
    ratio = im.shape[0] / 500.0
    orig = im.copy()
    im = imutils.resize(im, height = 500)
    kernel=np.ones((5,5))
    #closing = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    #gradient = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bin = cv2.threshold(gray,120,255,1) # inverted threshold (light obj on dark bg)
    bin = cv2.dilate(bin, None)  # fill some holes
    bin = cv2.dilate(bin, None)
    bin = cv2.erode(bin, None)   # dilate made our shape larger, revert that
    bin = cv2.erode(bin, None)
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rc = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rc)
    for p in box:
        pt = (p[0],p[1])
        print (pt)
        cv2.circle(im,pt,5,(200,0,0),2)
    c=imutils.resize(im,height=50)
    cv2.imwrite("coordinates.jpg", im)

    cv2.circle(image, (215, 115), 5, (0, 0, 255), -1)
    cv2.circle(image, (510, 113), 5, (0, 0, 255), -1)
    cv2.circle(image, (172, 495), 5, (0, 0, 255), -1)
    cv2.circle(image, (520, 495), 5, (0, 0, 255), -1)
    pts1 = np.float32([[215, 115], [510, 113], [172, 495], [520, 495]])
    pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(im, matrix, (500, 600))
    cv2.imwrite("Frame.jpg", image)
    cv2.imwrite("Perspectivetransformation.jpg", result)


def enhance_img(image1):
    enh_bri = ImageEnhance.Brightness(image1)
    brightness = 1
    image_brightened = enh_bri.enhance(brightness)
    #image_brightened.save("bright.jpg")

    image1 = image_brightened
    enh_col = ImageEnhance.Color(image1)
    color = 1.5
    image_colored = enh_col.enhance(color)
    #image_colored.save("color.jpg")

    image1 = image_colored
    enh_con = ImageEnhance.Contrast(image1)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    #image_contrasted.save("contrast.jpg")

    image1 = image_contrasted
    enh_sha = ImageEnhance.Sharpness(image1)
    sharpness = 1.0
    image_sharped = enh_sha.enhance(sharpness)
    image_sharped.save("/var/www/html/python/sharp.jpg")




blurr=blur(image);
print(blurr)

if(blurr==0):
    coordinates(image);
    image1 = Image.open("./paper12.jpg")
    enhance_img(image1);




    


