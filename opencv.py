#!/usr/bin/env python
#!-*- coding: utf8 -*-

#https://robotos.in/uroki/obnaruzhenie-i-raspoznavanie-litsa-na-python
#https://github.com/opencv/opencv/tree/master/data/haarcascades

import numpy as np
import cv2
import time
print cv2.__version__

#faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
#faceCascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

i = 0
padding = 55
while(True):  
    ret, frame = cap.read()
    i += 1
    if i % 3 != 0:
      continue
    
    #frame = cv2.flip(frame, -1)
    #cv2.imshow('frame', frame)
    
    #mask = cv2.inRange(frame, (100,100,100), (255,255,255) )
    #cv2.imshow('mask', mask)    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', gray)
    

    faces = faceCascade.detectMultiScale( frame, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
    for i,(x,y,w,h) in enumerate(faces):
	print x,y,w,h
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0), 2)
        roi_gray = gray[y-padding:y+h+padding, x-padding:x+w+padding]
        try:
	  cv2.imshow('roi_gray_'+str(i), roi_gray)
	except cv2.error as e:
	  print e

    cv2.imshow('video', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    #time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()


# https://www.youtube.com/watch?v=lDxyGKGVq3s - хитрожопая сеть
# https://www.youtube.com/watch?v=gB53a2KzpKE - t.A.T.u voice

