import cv2
import numpy as np
import face_recognition 
import os
from datetime import datetime
import pickle

path = 'photos/'
images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist: 
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# yuz ma'lumotlarini "encoded_faces.pickle" fayidan o'qib olish
with open('encoded_faces.pickle', 'rb') as f:
    encoded_face_train = pickle.load(f)


attendance_file = 'Attendance.csv'

def markAttendance(name):
    
    # opendoor()
    with open(attendance_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if name in line:
                return  # Name already exists in file, no need to add it again
        
    with open(attendance_file, 'a') as f:
        now = datetime.now()
        time = now.strftime('%I:%M:%S:%p')
        date = now.strftime('%d-%B-%Y')
        f.write(f"{name},{time},{date}\n")

cap  = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper().lower()
            y1,x2,y2,x1 = faceloc
            y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break