import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model & labels
model = load_model("emotion_model.h5")
labels = np.load("labels.npy", allow_pickle=True).item()
labels = {v: k for k, v in labels.items()}

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Read image
img = cv2.imread("test_image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:
    print("No face detected!")
    exit()

for (x, y, w, h) in faces:
    face = gray[y:y+h, x:x+w]  # crop
    face = cv2.resize(face, (48,48))
    face = face.reshape(1,48,48,1) / 255.0

    pred = model.predict(face)
    emotion = labels[np.argmax(pred)]

    print("Predicted Emotion:", emotion)
