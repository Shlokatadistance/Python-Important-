from PIL import Image
import pytesseract

im=Image.open("./test.jpg")
#pytesseract.pytesseract.tesseract_cmd =r'C:\Program Files\Tesseract-OCR'
text=pytesseract.image_to_string(im, lang='eng')
print(text)

