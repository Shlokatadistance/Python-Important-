from PIL import Image
im = Image.open('reference.jpg')
pix = im.load()
print (im.size)
print(pix[100,100])
