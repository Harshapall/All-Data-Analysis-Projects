#!/usr/bin/env python
# coding: utf-8

import cv2
import os
import numpy as np
from keras.models import load_model

os.chdir("E:\\MY_BLOCKS\\Facial_Captures\\models")

# Load emotion detection model
emotion_model = load_model('E:\\MY_BLOCKS\\Facial_Captures\\fer2013_mini_XCEPTION.102-0.66.hdf5')


def detectFace(net, frame, confidence_threshold=0.7):
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    frameOpencvDNN = frame.copy()
    frameHeight = frameOpencvDNN.shape[0]
    frameWidth = frameOpencvDNN.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDNN, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            face = frame[y1:y2, x1:x2]
            cv2.rectangle(frameOpencvDNN, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
            faceBoxes.append((x1, y1, x2, y2))

            # Predict emotion
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (64, 64))  # Resize to match the input shape of the model
            face = face / 255.0
            face = np.reshape(face, (1, 64, 64, 1))  # Add batch dimension
            emotion_prediction = emotion_model.predict(face)
            emotion_label = emotion_labels[np.argmax(emotion_prediction)]

            #cv2.putText(frameOpencvDNN, f'Emotion: {emotion_label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       # (0, 255, 255), 2, cv2.LINE_AA)

    return frameOpencvDNN, faceBoxes


# Initialize face detection model
faceProto = 'opencv_face_detector.pbtxt'
faceModel = 'opencv_face_detector_uint8.pb'
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Load age and gender detection models
ageProto = 'age_deploy.prototxt'
ageModel = 'age_net.caffemodel'
ageNet = cv2.dnn.readNet(ageModel, ageProto)

genderProto = 'gender_deploy.prototxt'
genderModel = 'gender_net.caffemodel'
genderNet = cv2.dnn.readNet(genderModel, genderProto)

genderList = ['Male', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Initialize video capture
video = cv2.VideoCapture(0)
padding = 20

while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg, faceBoxes = detectFace(faceNet, frame)

    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        x1, y1, x2, y2 = faceBox

        face = frame[max(0, y1 - padding):min(y2 + padding, frame.shape[0] - 1),
               max(0, x1 - padding):min(x2 + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)

        # Predict gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Predict age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # Display gender and age on the frame
        cv2.putText(resultImg, f'Gender: {gender}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(resultImg, f'Age: {age}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Detecting age, gender", resultImg)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
