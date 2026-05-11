Emotion Detection using CNN (NNDL Project)

Project Overview This project is developed as part of the Neural Networks and Deep Learning (NNDL) course.
The aim is to train a Convolutional Neural Network (CNN) to classify facial expressions into 7 different emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.

Project Structure NNDL Project/

data/train/<7 emotion folders>
data/test/<7 emotion folders>
train.py (CNN training script)
predict.py (Predict emotion for a single image)
emotion_model.h5 (Saved trained model)
labels.npy (Saved label dictionary)
training_graphs.png (Accuracy & loss graph)
test_image.jpg (Test image)
README.md (Documentation)
Installation & Setup

Create Conda Environment: conda create -n nndl python=3.9 -y conda activate nndl

Install Required Libraries: pip install tensorflow==2.10 pip install numpy==1.23.5 pip install matplotlib pip install opencv-python pip install scikit-learn

Dataset Use FER-2013 or any 7-class facial emotion dataset. Folder structure must be: data/train// data/test//

Training the Model Run: python train.py

This will:

Train a CNN
Validate model
Save emotion_model.h5
Save labels.npy
Generate accuracy/loss graphs
Predicting Emotion for a New Image Place your test image as test_image.jpg Run: python predict.py Output example: Predicted Emotion: happy

Model Performance Training Accuracy: ~70–75% Validation Accuracy: ~55–60%

Technologies Used

Python
TensorFlow / Keras
OpenCV
NumPy
Matplotlib
CNN
Applications

Emotion Classification
Sentiment Analysis
Academic NNDL Project
Limitations

Works best on cropped face images
Might misclassify subtle expressions
No real-time webcam detection
No MobileNetV2 upgrade included
Author M. N. Niharika Shree NNDL Project – 2025

