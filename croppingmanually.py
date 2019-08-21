import cv2
import imutils

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
ref_point = []
cropping = False

def shape_selection(event, x, y, flags, param):
	# grab references to the global variables
	global ref_point, cropping

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		ref_point = [(x, y)]
		cropping = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		ref_point.append((x, y))
		cropping = False

		# draw a rectangle around the region of interest
		cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
		cv2.imshow("image", image)

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True, help="Path to the image")
#args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread('reference_1.jpg')
#use it for white backgrounds
image = imutils.resize(image,height = 500)
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", shape_selection)

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()

	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break

# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(ref_point) == 2:
        xStart = ref_point[0][1]
        xEnd = ref_point[1][1]
        yStart = ref_point[1][0]
        yEnd = ref_point[0][0]
        if (xStart > xEnd):
            temp = xStart
            xStart = xEnd
            xEnd = temp
        if (yStart > yEnd):
            temp = yStart
            yStart = yEnd
            yEnd = temp
        cv2.rectangle(image, (ref_point[0][0],ref_point[0][1]),(ref_point[1][0],ref_point[1][1]),(0,0,0),2)
        nameImg = ("crop_img_{}_{}_{}_{}_{}.jpg").format(image,xStart, xEnd, yStart,yEnd)
        crop_img = image[xStart:xEnd,yStart:yEnd]
        
        
             
        #crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
        image = imutils.resize(image,height = 500)
        cv2.imshow("crop_img", crop_img)
        cv2.waitKey(0)
# close all open windows
cv2.destroyAllWindows()
"""
xStart = ref_point[0][1]
xEnd = ref_point[1][1]
yStart = ref_point[1][0]
yEnd = ref_point[0][0]
if (xStart > xEnd):
    temp = xStart
    xStart = xEnd
    xEnd = temp
if (yStart > yEnd):
    temp = yStart
    yStart = yEnd
    yEnd = temp
cv2.rectangle(image, (ref_point[0][0],ref_point[0][1]),(ref_point[1][0],ref_point[1][1]),(0,0,0),2)
nameImg = ("crop_img_{}_{}_{}_{}_{}.jpg").format(symbol, xStart, xEnd, yStart,yEnd)
crop_img = original[xStart:xEnd,yStart:yEnd]
"""

