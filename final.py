from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import urllib
from PIL import Image
from PIL import ImageEnhance
import boto3

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required = True)
args = vars(ap.parse_args())
#image = cv2.imread(args["image"])

url_response = urllib.urlopen(args["image"])
img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
image = cv2.imdecode(img_array, -1)

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

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped

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
    gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    print(cnt)
    cv2.imwrite('croppedimg.jpg',crop)
    #cropped= Image.fromarray(crop)
    cropped=Image.open('croppedimg.jpg')
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
    image_sharped.save("final_output.jpg")

    client = boto3.client('s3', region_name='ap-south-1')
    client.upload_file('/var/www/html/python/' + 'final_output.jpg', 'dockboyz', 'uploads/after/{}'.format('final_output.jpg'))
 



blurr=blur(image);
print(blurr)


if(blurr==0):
    print("STEP 1: Edge Detection")
    #image = cv2.imread(args["image"])
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    
    print("STEP 2: Find contours of paper")
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

        if len(approx)!=4:
            print("STEP 4: Mask and Enhance")
            mask(orig);
            exit();
    
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

    print("STEP 3: Apply perspective transform")
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    #cv2.imshow("Original", imutils.resize(orig, height = 650))
    cv2.imwrite("crop.jpg", imutils.resize(warped, height = 650))

    print("STEP 4: Enhance Image")
    cropped=Image.open('crop.jpg')
    enhance_img(cropped);