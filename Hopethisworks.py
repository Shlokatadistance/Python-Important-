import cv2
import os
import imutils
import numpy as np
from skimage.filters import threshold_local
from skimage import data
import skimage
from PIL import ImageEnhance
from PIL import Image

#from pyimagesearch.transform import four_point_transform
#four point transform is used as a function in here
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped
def blurrchecker(image):
    return cv2.Laplacian(image,cv2.CV_64F).var()

image = cv2.imread('test1.jpg')
#checks the value of blurring in the image, is pretty accurate

if (blurrchecker(image) < 80.0):
    print("Please re-take the image")
    print(blurrchecker(image))
    #sys.exit()
    
else:
    print(blurrchecker(image))
    ratio = image.shape[0]/500.0
    #the ratio thing is perfectly alright
    #whenever you get the nonetype object has no attribute ' error, check whether the input is correct
    #or not
    orig = image.copy()
    image= imutils.resize(image,height = 500)
    #it basically just determines the size at which the image is displayed 

    #converting into grayscale and finding the edges
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    retval,threshold = cv2.threshold(image,30,255,cv2.THRESH_BINARY)
    #retval,threshold = cv2.threshold(image,30,255,cv2.ADAPTIVE_THRESH_MEAN_C)
    kernel = np.ones((5,5),np.uint8)
    #us the eroded method if the thing cannot find 4 edges
    #eroded = cv2.erode(image,kernel,iterations = 5)
    #5 is ideal, 6 works too, doesnt work over 6
    #also here, the thresh values need to be refreshed

    grey = cv2.GaussianBlur(image,(5,5),0)
    #even this one, the gray isnt working properly
    #the gaussian blur is creating issues while detecting the edges
    #the purpose of the blur is to remove the noise in the picture
    #two techniques can be used, erosion and dilation

    edged = cv2.Canny(grey,75,200)
    #i have either used gray or image
    #edge detected image
    #in the reference_11, the algorithm removed most of the text from inside the form
    cv2.imshow('image',image)
    cv2.imshow('edges',edged)
    #cv2.imwrite('edges1.jpg',edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    def rotation(image,angle):
        h,w = image.shape[:2]
        center = (w/2,h/2)
        #angle90 = 90
        #angle270 = 270
        #angle180 = 180
        scale = 1.0
        rotated_image = imutils.rotate_bound(image,angle)
        #imutils.rotate_bound ensures that the rotated picture does not go out of frames
        #M = cv2.getRotationMatrix2D(center,angle,scale)
        #rotated_image = cv2.warpAffine(image,M,(w,h))
        #cv2.imshow('rotation',rotated_image)
        return rotated_image

    #step 2 , finding the contours in the picture
    #we assueme that the largest triangle in the contour is the paper with its four edges
    #finding contours in the edged image
    cnts = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #retr list or retr external
    #also here, the modification needs to be done
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts,key = cv2.contourArea,reverse = True)[:5]
    #changing countour area to contour length
    #there is no parameter such as contourLength
    #now looping over the contours
    idx = 0
    for c in cnts:
        peri = cv2.arcLength(c,True)
       #using epsilon method 
        approx = cv2.approxPolyDP(c,0.02 *peri,True)
        x,y,w,h = cv2.boundingRect(c)
        if w > 50 and h > 50:
            idx += 1
            new = image[y:y+h,x:x+w]
            #cv2.imshow('cropped',new)
        #swapping between true and false,not working
        #there needs to be made a modification here

        if len(approx) == 4:
            screenCnt = approx
            break

        else:
                
            print("no four vertices have been properly found")
            
            def clahe(img, clip_limit=2.0, grid_size=(8,8)):
                    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
                    return clahe.apply(img)
                    
                    
                #clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
                 #return clahe.apply(img)



            src = cv2.imread('test5.jpg')
            ratio = src.shape[0] / 500.0
            #ratio = img.shape[0]/400.0

            orig = src.copy()
            src = imutils.resize(src, height = 500)
            kernel=np.ones((5,5))
            score = cv2.Laplacian(src, cv2.CV_64F).var()
            array = (np.random.rand(100,200)*256).astype(np.uint8)

            if score > 300:

                #print ("Not Blur")
                cv2.imshow("src", src)
                hsv = cv2.cvtColor(src.copy(), cv2.COLOR_BGR2HSV)
                lower_blue = np.array([0, 0, 120])
                upper_blue = np.array([180, 38, 255])
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                result = cv2.bitwise_and(src, src, mask=mask)
                b, g, r = cv2.split(result)
                g = clahe(g, 5, (3, 3))
                img_blur = cv2.blur(g, (9, 9))
                img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 51, 2)
                cv2.imshow("img_th", img_th)

                _,contours, hierarchy = cv2.findContours(img_th,
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
                        if brightness > max_brightness:
                            brightest_rectangle = rect
                            max_brightness = brightness
                        cv2.imshow("mask", mask)
                        print(rect)
                        break
                        exit(1)
                        

                if w*h > 40000:
                    x, y, w, h = brightest_rectangle
                    #only detecting the most obvious rectangle
                    #the coordinates displayed are the coordinates to crop
                    cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 1)
                    cv2.imwrite('mask.png',mask)
                    img = cv2.imread('mask.png')
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
                    _, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cnt = contours[0]
                    x,y,w,h = cv2.boundingRect(cnt)
                    print(w*h)
                    crop = img[y:y+h,x:x+w]
                    print(cnt)
                    cv2.imwrite('crop.jpg',crop)
                    cv2.imshow('crop',crop)
                    #im = crop.copy()
                    #cv2.imshow('im',im)
                    c=np.mean(img)
                    print(c)
                    cropped= Image.fromarray(crop)
                    #cropped.save(r'C:\Users\AdityaJoshi\Desktop\zapfin\e.jpg')
                    #e=cv2.imread(r'C:\Users\AdityaJoshi\Desktop\zapfin\e.jpg')
                    #e = cv2.cvtColor(e, cv2.COLOR_BGR2RGB)
                    #cv2.imshow('e',e)
                    #temp=cv2.imread(r'C:\Users\AdityaJoshi\Desktop\zapfin\crop.jpg')
                    b=np.mean(cropped)
                    print(b)
                    if b < 150:
                        x=1
                    else:
                        x=1.2
                    enh_bri = ImageEnhance.Brightness(cropped)
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
                    image_sharped.save('e1.jpg')
                    e1=cv2.imread('e1.jpg')
                    cv2.imshow('e1',e1)
                    break
                    #image_sharped.save(r"C:\Users\AdityaJoshi\Desktop\zapfin\e.jpg")
                    #e=cv2.imread(r"C:\Users\AdityaJoshi\Desktop\zapfin\e.jpg")
                    #cv2.imshow('e',e)
                else:
                    print("Document cannot be detected.")
                    print("Take photo again.")
                    print("THe photo cannot be uploaded to this s")
            else:
                print ("Blur")
                print("Blur score:",score)

















            
            
    #show the contour of the paper
    #cv2.boxpoint(cnts)
    cv2.drawContours(image,[screenCnt],-1,(0,255,0),2)
    #cv2.imwrite('outline1.jpg',image)
    cv2.imshow('outline',image)
     
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #the basic purpose of the below steps is to create a top down view of the image
    #which it does

    warped = four_point_transform(orig,screenCnt.reshape(4,2) *ratio)
    #warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
    #warped = skimage.exposure.rescale_intensity(warped, out_range = (0, 255))
    #block_size = 35
    
    #T = threshold_local(warped,11,offset = 10,method = "gaussian")
    #warped = (warped > T).astype("uint8") * 255
    warped_new = imutils.resize(warped,height =  500)
    #warped_new = exposure.rescale_intensity(warped_new, out_range = (0, 255))
    #warped_new = cv2.cvtColor(warped_new,cv2.COLOR_GRAY2RGB)
    cv2.imshow('newscanned',warped_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    a = input("do you need to rotate the image?")
    if a == "Yes":
        ang = int(input("Enter the rotation angle:"))
        image_new = rotation(warped_new,ang)
        cv2.imshow('final output',image_new)
    else:
        print("You didnt say anything else")
#there is actually no way to convert a grayscale image back to rgb

#if (numpy.ravel == [0,0,0]):
  #  print("the background is white")
#elif (numpy.ravel == [0,255,0] or numpy.ravel == [255,0,0] or numpy.ravel[0,0,255]):
  #  print("The backgorund is not clear, reclick the picture")
#else:
  #  print("The backgorund is black")

    
 
