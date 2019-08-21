import cv2
import imutils
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import argparse
import urllib


ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required = True)
args = vars(ap.parse_args())
#image = cv2.imread(args["image"])
url_response = urllib.urlopen(args["image"])
img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
image = cv2.imdecode(img_array, -1)


def clahe(img, clip_limit=2.0, grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)

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

def mask(image):
    src = image
    #ratio = src.shape[0] / 500.0
    orig = src.copy()
    #src = imutils.resize(src, height = 500)

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
    img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 51, 2)

    contours, hierarchy = cv2.findContours(img_th,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    #Filter the rectangle by choosing only the big ones
    #and choose the brightest rectangle as the document
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
            #cv2.imshow("mask", mask)
            print(rect)
            #cv2.waitKey(0)
            break

    x, y, w, h = brightest_rectangle
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 1)
    #cv2.imshow("canvas", canvas)
    cv2.imwrite('maskedimg.jpg',mask)
    mask_img=mask
    #cv2.imwrite(r'C:\Users\AdityaJoshi\Desktop\zapfin\mask.png',mask)
    #cv2.waitKey(0)
    #img = cv2.imread('mask.png')
    img=mask_img
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    print(cnt)
    cv2.imwrite('croppedimg.jpg',crop)
    cropped= Image.fromarray(crop)
    enhance_img(cropped);
    #cv2.imshow('crop',crop)
    #return crop


def enhance_img(image1):
    enh_bri = ImageEnhance.Brightness(image1)
    brightness = 1.2
    image_brightened = enh_bri.enhance(brightness)
    #image_brightened.save("bright.jpg")

    image1 = image_brightened
    enh_col = ImageEnhance.Color(image1)
    color = 1.6
    image_colored = enh_col.enhance(color)
    #image_colored.save("color.jpg")

    image1 = image_colored
    enh_con = ImageEnhance.Contrast(image1)
    contrast = 2
    image_contrasted = enh_con.enhance(contrast)
    #image_contrasted.save("contrast.jpg")

    image1 = image_contrasted
    enh_sha = ImageEnhance.Sharpness(image1)
    sharpness = 1.3
    image_sharped = enh_sha.enhance(sharpness)
    image_sharped.save("sharp.jpg")

blurr=blur(image);
print(blurr)

if(blurr==0):
    mask(image);
    #image1 = Image.open('./crop2.png')
    #enhance_img(image1);



    


