import cv2
import imutils
import numpy as np
from PIL import Image
from PIL import ImageEnhance

image = Image.open("./paper12.jpg")
enh_bri = ImageEnhance.Brightness(image)
brightness = 1
image_brightened = enh_bri.enhance(brightness)
#image_brightened.save("bright.jpg")

image = image_brightened
enh_col = ImageEnhance.Color(image)
color = 1.5
image_colored = enh_col.enhance(color)
#image_colored.save("color.jpg")

image = image_colored
enh_con = ImageEnhance.Contrast(image)
contrast = 1.5
image_contrasted = enh_con.enhance(contrast)
#image_contrasted.save("contrast.jpg")

image = image_contrasted
enh_sha = ImageEnhance.Sharpness(image)
sharpness = 1.0
image_sharped = enh_sha.enhance(sharpness)
image_sharped.save("/var/www/html/python/enhanced_img.jpg")