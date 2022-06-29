# import the necessary packages
from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image
import requests
import os
import glob

direction = 'C:\\Users\\vnhnntt\\Truong data\\learn\\Python\\Udemy - Python - Data Scient mega course\\Direction'
subdir = 'C:\\Users\\vnhnntt\\Truong data\\learn\\Python\\Udemy - Python - Data Scient mega course\\Direction\\image'
#os.makedirs(direction)
#os.makedirs(subdir)

# Open image and display
template = Image.open('C:\\Users\\vnhnntt\\Truong data\\learn\\Python\\Udemy - Python - Data Scient mega course\\pic\\template.jpg')
supply = Image.open('C:\\Users\\vnhnntt\\Truong data\\learn\\Python\\Udemy - Python - Data Scient mega course\\pic\\supply.jpg')

# The file format of the source file.
print("Template image format : ",template.format) 
print("Supply image format : ",supply.format)

# Image size, in pixels. The size is given as a 2-tuple (width, height).
print("Template image size : ",template.size)
print("Supply image size : ",supply.size)

# Resize Image
template = template.resize((250, 160))
#print(original.size)
template.save('C:\\Users\\vnhnntt\\Truong data\\learn\\Python\\Udemy - Python - Data Scient mega course\\Direction\\image\\template.png')#Save image
supply = supply.resize((250,160))
#print(tampered.size)
supply.save('C:\\Users\\vnhnntt\\Truong data\\learn\\Python\\Udemy - Python - Data Scient mega course\\Direction\\image\\supply.png')#Saves image

#template.show()
#supply.show()

# load the two input images
template = cv2.imread('C:\\Users\\vnhnntt\\Truong data\\learn\\Python\\Udemy - Python - Data Scient mega course\\Direction\\image\\template.png')
supply = cv2.imread('C:\\Users\\vnhnntt\\Truong data\\learn\\Python\\Udemy - Python - Data Scient mega course\\Direction\\image\\supply.png')

# Convert the images to grayscale
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
supply_gray = cv2.cvtColor(supply, cv2.COLOR_BGR2GRAY)

#cv2.imshow('Original image', template_gray) #to display original image which is not convert to gray
#cv2.imshow('Gray image', template_gray)
#cv2.imshow('Gray image', supply_gray)

# Compute the Structural Similarity Index (SSIM) between the two images, ensuring that the difference image is returned
(score, diff) = structural_similarity(template_gray, supply_gray, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# Calculating threshold and contours 
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
    # applying contours on image
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(template, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(supply, (x, y), (x + w, y + h), (0, 0, 255), 2)

#Diplay original image with contour
#print('Original Format Image')
#Image.fromarray(template).show()

#Diplay tampered image with contour
print('Tampered Image')
Image.fromarray(supply).show()
