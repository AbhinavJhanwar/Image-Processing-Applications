# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:41:03 2019

@author: abhinav.jhanwar
"""

''' detects dark objects '''

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time

# read the image with object to be tracked
frame = cv2.imread('object.jpg')

vc = cv2.VideoCapture(0)

while True:
    
    ret, frame = vc.read()
    
    # resize
    #frame = imutils.resize(frame, height=400)
    
    # blur it to remove noise, convert to gray scale and find edges
    blurred = cv2.GaussianBlur(frame, (9, 9), 0)
    #gray =  cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(blurred, (10,10,10), (100,100,100))
    mask = cv2.erode(mask, None, iterations=15)
    mask = cv2.dilate(mask, None, iterations=15)
    #mask = cv2.Canny(blurred, 30, 150)
    #mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow("frame", mask)
    #cv2.waitKey(1)
    
    # find contours
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    output = frame.copy()
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
        #print(ar, w, h)
        
        if w>250 and h>200 and ar<1.5 and ar>0.5:
            cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
            print(ar, w, h)
        
    cv2.imshow("Contours", output)
        #cv2.waitKey(0)
        
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        vc.release()
        cv2.destroyAllWindows()
        break
