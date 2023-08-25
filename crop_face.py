import face_recognition
import os
import cv2

image_dir = 'photos/' # To'plangan rasmlar joylashgan yo'l
output_dir = 'faces/' # qirqib olingan yuz rasmlarini saqlash uchun

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('jpg') or file.endswith('jpeg') or file.endswith('png'):
            path = os.path.join(root, file)
            img = face_recognition.load_image_file(path)
            faces = face_recognition.face_locations(img)

            for (top, right, bottom, left) in faces:
                face_img = img[top:bottom, left:right]
                resized_img = cv2.resize(face_img, (128, 128))
                
                filename = os.path.splitext(file)[0] + '.jpg'
                cv2.imwrite(os.path.join(output_dir, filename), resized_img)
