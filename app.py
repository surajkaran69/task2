import streamlit as st
import torch
from PIL import Image
import numpy as np
import pickle
import pandas as pd
from torchvision import transforms
from torch import nn
from io import BytesIO
import datetime
import os

# Load the trained model and label encoder
model_path = 'emotion_detection_model.pkl'
label_encoder_path = 'label_encoder.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Helper function to predict emotion
def predict_emotion(image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        emotion = label_encoder.inverse_transform([predicted.item()])
    return emotion[0]

# Streamlit UI
st.title('Attendance and Emotion Detection System')

# Time check to only allow detection between 9:30 AM and 10:00 AM
current_time = datetime.datetime.now().time()
start_time = datetime.time(9, 30)
end_time = datetime.time(10, 0)

if start_time <= current_time <= end_time:
    # Image Upload
    uploaded_file = st.file_uploader("Upload an image of the student", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict emotion
        predicted_emotion = predict_emotion(image)
        st.write(f"Predicted Emotion: {predicted_emotion}")

        # Log student presence and emotion with timestamp
        if st.button("Mark Attendance"):
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            student_data = {
                "Timestamp": current_time,
                "Emotion": predicted_emotion,
                "Student": "Present"  # Assume the student is present when detected
            }

            # Load the existing attendance CSV, if available
            if os.path.exists("attendance_log.csv"):
                attendance_df = pd.read_csv("attendance_log.csv")
            else:
                attendance_df = pd.DataFrame(columns=["Timestamp", "Emotion", "Student"])

            # Append new data
            attendance_df = attendance_df.append(student_data, ignore_index=True)
            attendance_df.to_csv("attendance_log.csv", index=False)

            st.success("Attendance marked successfully!")

            # Download link for the CSV
            csv_file = BytesIO()
            attendance_df.to_csv(csv_file, index=False)
            csv_file.seek(0)
            st.download_button(label="Download Attendance Log", data=csv_file, file_name="attendance_log.csv", mime="text/csv")
else:
    st.warning("Attendance system is only available from 9:30 AM to 10:00 AM.")
