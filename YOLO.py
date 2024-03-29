#importing libraries

import cv2 as cv
import numpy as np


#Setting conf,nms thresholds, etc
conThreshold = 0.25
nmsThreshold = 0.40

#shaping output
inpWidth = 416   #input Width
inpHeight = 416  #input Height


def getOutputsNames(net):
    layerNames = net.getLayerNames()
    return [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers


def postprocess(frame,out):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classID = []
    confidence = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores

    if confidence > threshold:

        centerX = int(detection[0]*frameWidth)
        centerYv= int(detection[1]*frameWidth)

        width = int(detection[2]*frameHeight)
        height = int(detection[3]*frameHeight)

        left = int(centerX-width/2)
        top = int(centerY-height/2)


#Reading the name files

classesFile = "coco.names"
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

net.setInput(blob)
outs = net.forward(getOutputsNames(net))

postprocess(frame,outs)

cv.imshow(winName,frame)
