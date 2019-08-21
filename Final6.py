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
url_response = urllib.request.urlopen(args["image"])
img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
image = cv2.imdecode(img_array, -1)

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def clahe(img, clip_limit=2.0, grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)

def mask(image):
    src = image
    orig = src.copy()
    hsv = cv2.cvtColor(src.copy(), cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 120])
    upper_blue = np.array([180, 38, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(src, src, mask=mask)
    b, g, r = cv2.split(result)
    g = clahe(g, 5, (3, 3))


    img_blur = cv2.blur(g, (9, 9))
    img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 51, 2)

    contours, hierarchy = cv2.findContours(img_th,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
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
            print(rect)
            break

    x, y, w, h = brightest_rectangle
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 1)
    cv2.imwrite('maskedimg.jpg',mask)
    gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = image[y:y+h,x:x+w]
    print(cnt)
    cv2.imwrite('croppedimg.jpg',crop)
    cropped=Image.open('croppedimg.jpg')
    enhance_img(cropped);


def enhance_img(image1):
    print('Step 5. Enhance')
    enh_bri = ImageEnhance.Brightness(image1)
    brightness = 1.2
    image_brightened = enh_bri.enhance(brightness)

    image1 = image_brightened
    enh_col = ImageEnhance.Color(image1)
    color = 1.6
    image_colored = enh_col.enhance(color)

    image1 = image_colored
    enh_con = ImageEnhance.Contrast(image1)
    contrast = 2
    image_contrasted = enh_con.enhance(contrast)

    image1 = image_contrasted
    enh_sha = ImageEnhance.Sharpness(image1)
    sharpness = 1.3
    image_sharped = enh_sha.enhance(sharpness)
    image_sharped.save("output/demo_output.jpg")

    #client = boto3.client('s3', region_name='ap-south-1')
    #client.upload_file('/var/www/html/python/' + 'demo_output.jpg', 'dockboyz', 'uploads/after/{}'.format('demo_output.jpg'))

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
    

    return warped, maxHeight*maxWidth

def blurrchecker(image):
    return cv2.Laplacian(image,cv2.CV_64F).var()

#image = cv2.imread('paper23.png')

print("Step 1. Check blur score")
if (blurrchecker(image) < 80.0):
    print("Please re-take the image")
    print(blurrchecker(image))
else:
    print(blurrchecker(image))
    ratio = image.shape[0]/500.0
    orig = image.copy()
    image= imutils.resize(image,height = 500)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    retval,threshold = cv2.threshold(image,30,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)

    grey = cv2.GaussianBlur(image,(5,5),0)
    print("step 2. Edge Detection")
    edged = cv2.Canny(grey,75,200)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("step 3. Find contours")
    cnts = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts,key = cv2.contourArea,reverse = True)[:5]

    idx = 0
    for c in cnts:
        peri = cv2.arcLength(c,True)

        approx = cv2.approxPolyDP(c,0.02 *peri,True)
        x,y,w,h = cv2.boundingRect(c)
        if w > 50 and h > 50:
            idx += 1
            new = image[y:y+h,x:x+w]


        if len(approx) == 4:
            screenCnt = approx
            break
        if len(approx) != 4:
            print("Step 4. Mask")
            mask(orig);
            exit();

    cv2.drawContours(image,[screenCnt],-1,(0,255,0),2)


     
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('Step 4. Perspective Transform')
    warped,a = four_point_transform(orig,screenCnt.reshape(4,2) *ratio)

    if len(approx) ==4 and a<40000:
        mask(orig);
        exit();

    warped_new = imutils.resize(warped,height =  500)

    cv2.imwrite('newscanned.jpg',warped_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cropped=Image.open('newscanned.jpg')
    enhance_img(cropped);