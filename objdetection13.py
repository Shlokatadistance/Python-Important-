###### BLUR  #######

from PIL import Image,ImageFilter
image=Image.open(r'C:\Users\AdityaJoshi\Desktop\zapfin\paper.jpg')
cropped_image=image.crop((105,25,330,160))
blurred_image=cropped_image.filter(ImageFilter.GaussianBlur(5))
image.paste(blurred_image,(105,25,330,160))
image.show()
