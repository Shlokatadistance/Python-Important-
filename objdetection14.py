from PIL import Image
img = Image.open(r'C:\Users\AdityaJoshi\Desktop\zapfin\shape.png')
img2 = img.crop((96, 90, 407, 470))
img2.show()
