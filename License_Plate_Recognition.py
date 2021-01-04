import numpy as np
import cv2
import imutils
import sys
import pytesseract
import pandas as pd
import time

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"


# read and resize image to the required size
image = cv2.imread('image.jpg')
image = imutils.resize(image, width=500)
cv2.imshow("Original Image", image)

# convert to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Conversion", gray)

# blur to reduce noise
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("Bilateral Filter", gray)

# perform edge detection
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("Canny Edges", edged)

cv2.waitKey(0)
cv2.destroyAllWindows()

# find contours in the edged image
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]

NumberPlateCnt = None 
count = 0
# loop over contours
for c in cnts:
	# approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if the approximated contour has four points, then assume that screen is found
    if len(approx) == 4:  
        NumberPlateCnt = approx 
        break

# mask the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
new_image = cv2.bitwise_and(image,image,mask=mask)
cv2.namedWindow("Final Image",cv2.WINDOW_NORMAL)
cv2.imshow("Final Image",new_image)

# configuration for tesseract
config = ('-l eng --oem 1 --psm 3')

# run tesseract OCR on image
text = pytesseract.image_to_string(new_image, config=config)

# data is stored in CSV file
raw_data = {'date':[time.asctime( time.localtime(time.time()))],'':[text]}
df = pd.DataFrame(raw_data)
df.to_csv('data.csv',mode='a')

# print recognized text
print(text)

cv2.waitKey(0)
cv2.destroyAllWindows()