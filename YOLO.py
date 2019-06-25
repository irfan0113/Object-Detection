#importing libraries

import cv2 as cv
import numpy as np


#Setting conf,nms thresholds, etc
conThreshold = 0.25
nmsThreshold = 0.40

#shaping output
inpWidth = 416   #input Width
inpHeight = 416  #input Height

#Reading the name files

classesFile = "coco.name"
classes = None

with open(classesFile,'rt')as f:
    classes=f.read().rstrip("\n").split("\n")

#reading the configs and weights

modelConf = "yolov3.cfg"
modelWeights = "yolov3.weights"


#Here we start openCV

net = cv.dnn.readNetFromDarknet(modelConf,modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DN_TARGET_CPU)

#naming window and resizing it
winName = "Object Detection With YOLO"

cv.namedWindow(winName,cv.WINDOW_NORMAL)
cv.resizeWindow(winName,1000,1000)


#Source of video
cap = cv.VideoCapture(0)

while cv.waitKey(1)<0:
    hasFrame, frame = cap.read()

blob = cv.dnn.blobFromImage(frame, 1/255,(inpWidth,inpHeight),[0,0,0],1,crop=False)
