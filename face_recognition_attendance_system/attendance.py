
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
from datetime import datetime

# Load model
model = load_model('models/final_model.h5')

# Labels
dir_path = "data/train"
labels = os.listdir(dir_path)

# Initialize attendance dictionary
attendance = {label: False for label in labels}

# Open webcam
cap = cv2.VideoCapture(0)

# Image preprocess function
def preprocess_face(face):
    face = cv2.resize(face, (224, 224))
    face = face / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    return face

print("Starting attendance...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        processed_face = preprocess_face(face_img)
        preds = model.predict(processed_face)
        label_index = np.argmax(preds)
        label = labels[label_index]
        confidence = preds[0][label_index]
        if confidence > 0.7 and not attendance[label]:
            attendance[label] = True
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Attendance marked for {label} at {now}")
            cv2.putText(frame, f"{label} Present", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('Attendance', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
