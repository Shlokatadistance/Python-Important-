import cv2
import numpy as np
import imutils
import argparse
import boto3
from PIL import ImageEnhance
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required = True)
args = vars(ap.parse_args())
#Now reading the image url
req = urllib.urlopen(args["image"])
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
src = cv2.imdecode(arr, -1)

def clahe(img, clip_limit=2.0, grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)
#used for histogram equaliazation, to change brightness and contrast

#src = cv2.imread(r'C:\Users\AdityaJoshi\Desktop\zapfin\zapfin.jpg')
ratio = src.shape[0] / 500.0
orig = src.copy()
src = imutils.resize(src, height = 500)
kernel=np.ones((5,5))
score = cv2.Laplacian(src, cv2.CV_64F).var()
#src=cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)

if score > 10:
    #print ("Not Blur")
    #cv2.imshow("src", src)
    hsv = cv2.cvtColor(src.copy(), cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 120])
    upper_blue = np.array([180, 38, 255])
    #the image is split into different colour regions by converting to hsv
    #the rectangle(required part) is then masked, i.e blackened
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(src, src, mask=mask)
    #it is masked at this step
    b, g, r = cv2.split(result)
    g = clahe(g, 5, (3, 3))
    img_blur = cv2.blur(g, (9, 9))
    img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 51, 2)
    
    #cv2.imshow("img_th", img_th)

    contours, hierarchy = cv2.findContours(img_th,
                                           cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    max_brightness = 0
    canvas = src.copy()
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        if w*h > 40000:
            mask = np.zeros(src.shape, np.uint8)
            mask[y:y+h, x:x+w] = src[y:y+h, x:x+w]
            brightness = np.sum(mask)
            #c
            if brightness > max_brightness:
                brightest_rectangle = rect
                max_brightness = brightness
            #cv2.imshow("mask", mask)
            print(rect)
            break

    if w*h > 40000:
        x, y, w, h = brightest_rectangle
        #in built function to find most significant rectangle,
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 1)
        #cv2.imwrite(r'C:\Users\AdityaJoshi\Desktop\zapfin\mask.png',mask)
        cv2.imwrite('mask.png',mask)
        #img = cv2.imread(r'C:\Users\AdityaJoshi\Desktop\zapfin\mask.png')
        img = cv2.imread('mask.png')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        print(w*h)
        crop = img[y:y+h,x:x+w]
        print(cnt)
        #cv2.imwrite(r'C:\Users\AdityaJoshi\Desktop\zapfin\crop.jpg',crop)
        cv2.imwrite('crop.jpg',crop)
        #cv2.imshow('crop',crop)
        im = crop.copy()
        #cv2.imshow('im',im)
        c=np.mean(img)
        #finding brightness of the image
        print(c)
        #cropped= Image.fromarray(crop)
        #cv2.imwrite(r'C:\Users\AdityaJoshi\Desktop\zapfin\e.jpg',crop)
        cv2.imwrite('e.jpg',crop)
        #e=Image.open(r'C:\Users\AdityaJoshi\Desktop\zapfin\e.jpg')
        e = Image.open(r'var\www\html\python\e.jpg')
        #e = cv2.cvtColor(e, cv2.COLOR_BGR2RGB)
        #cv2.imshow('e',e)
        #temp=cv2.imread(r'C:\Users\AdityaJoshi\Desktop\zapfin\crop.jpg')
        b=np.mean(e)
        print(b)
        if b < 150:
            x=1
        else:
            x=1.2
        enh_bri = ImageEnhance.Brightness(e)
        brightness = x
        image_brightened = enh_bri.enhance(brightness)
        enh_col = ImageEnhance.Color(image_brightened)
        color = x
        image_colored = enh_col.enhance(color)
        enh_con = ImageEnhance.Contrast(image_colored)
        contrast = x
        image_contrasted = enh_con.enhance(contrast)
        enh_sha = ImageEnhance.Sharpness(image_contrasted)
        sharpness = x
        image_sharped = enh_sha.enhance(sharpness)
        #image_sharped.save(r'C:\Users\AdityaJoshi\Desktop\zapfin\e1.jpg')
        image_sharped.save(r'\var\www\html\python\e1.jpg')
        
        #e1=cv2.imread(r'C:\Users\AdityaJoshi\Desktop\zapfin\e1.jpg')
        e1 = cv2.imread('e1.jpg')
        cv2.imwrite('e1.jpg',e1)
        client = boto3.client('s3', region_name='ap-south-1')

        client.upload_file('var/www/html/python/final_output.jpg', 'dockboyz/uploads', '.e1jpg')


        #image_sharped.save(r"C:\Users\AdityaJoshi\Desktop\zapfin\e.jpg")
        #e=cv2.imread(r"C:\Users\AdityaJoshi\Desktop\zapfin\e.jpg")
        #cv2.imshow('e',e)
    else:
        print("Document cannot be detected.")
        print("Take photo again.")
else:
    print ("Blur")
    print("Blur score:",score)
