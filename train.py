import cv2
import numpy as np
import face_recognition 
import os
from datetime import datetime
import pickle


path = 'faces/' # qirqib olingan yuz rasmlarini saqlash uchun
images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList


encoded_face_train = findEncodings(images)
with open('encoded_faces.pickle', 'wb') as f:
    pickle.dump(encoded_face_train, f)

with open('faces_names.pickle', 'wb') as f:
    pickle.dump(classNames, f)
