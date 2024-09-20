import cv2
import numpy as np
#import matplotlib.pyplot as plt
from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep
                
print("Loaded libs")
#Initializing camera connection
camera = PiCamera()
rawCapture = PiRGBArray(camera)
                
sleep(0.1)
                
camera.capture(rawCapture, format="bgr")
img = rawCapture.array
                
print("Took picture")
#IMPORTANT \/
camera.close()
                
#cv2.imshow('Captured', img)
                
#Blur to lose some of the noise
blur = cv2.GaussianBlur(img, (1,1),0)
                
#Creating grayscale images from base
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
#Using Canny to find edges
edges = cv2.Canny(blur, 50, 50)
                
kernel = np.ones((3,3),np.uint8)
edges = cv2.dilate(edges, kernel, iterations = 3 )
edges = cv2.erode(edges, kernel, iterations = 2 )
cv2.imshow('Edges', edges)
                
print("Finished pre-processing")
#Finding contours
contours_1, hierarchy_1 = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
print("Found contours")
#Getting base img size
img_w, img_h, _ = img.shape
area_fac = 0.001*img_h*img_w
area_max = 0.01*img_h*img_w
                
output = gray_rgb.copy()
#Filtering contours based on area and shape
for c in contours_1:
    rec = cv2.minAreaRect(c)
    (x, y), (width, height), angle = rec
    area = cv2.contourArea(c)
    box = cv2.boxPoints(rec)
    box = np.int0(box)
                
    if area>area_fac and height>0 and area<area_max:
        ratio = width/height
        #cv2.circle(output, (int(x), int(y)), 4, (0,0,255), -1)
        if ratio > 0.5 and ratio < 2:
            cv2.drawContours(output, [box], -1, (255, 0, 255), 1)
print("Filtered contours")
#Final result
cv2.imshow("Output", output)
cv2.imwrite("output_file.png", output)
print("Done")
                
cv2.waitKey()