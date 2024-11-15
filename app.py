import os
import cv2
import torch
import pandas as pd
from datetime import datetime
from torchvision import transforms
from PIL import Image
import pickle
import streamlit as st

# Paths
model_path = '/content/emotion.pkl'  # Path to your trained model
label_encoder_path = '/content/label_encoder.pkl'  # Path to your label encoder
output_csv_path = '/content/attendance.csv'  # Output attendance file

# Time Window
start_time = "09:30"
end_time = "10:00"

# Load the Model
with open(model_path, 'rb') as f:
    model = pickle.load(f)
model.eval()

# Load the Label Encoder
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Attendance Data
attendance_data = []

# Helper Function to Check Time
def is_within_time_window():
    now = datetime.now().time()
    start = datetime.strptime(start_time, "%H:%M").time()
    end = datetime.strptime(end_time, "%H:%M").time()
    return start <= now <= end

# Process Frame and Log Attendance
def process_frame(frame):
    global attendance_data

    # Convert OpenCV BGR to PIL RGB
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply Transformations
    input_tensor = transform(image).unsqueeze(0)

    # Get Predictions
    with torch.no_grad():
        outputs = model(input_tensor)
        student_idx, emotion_idx = outputs[0].argmax(1).item(), outputs[1].argmax(1).item()

    student_name = label_encoder.inverse_transform([student_idx])[0]
    emotion = label_encoder.inverse_transform([emotion_idx])[0]

    # Record Attendance
    if is_within_time_window():
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        attendance_entry = {
            'Name': student_name,
            'Emotion': emotion,
            'Timestamp': timestamp,
            'Status': 'Present'
        }

        # Avoid duplicate entries for the same student
        if not any(entry['Name'] == student_name for entry in attendance_data):
            attendance_data.append(attendance_entry)

    return student_name, emotion

# Save Attendance to CSV
def save_attendance():
    df = pd.DataFrame(attendance_data)
    df.to_csv(output_csv_path, index=False)
    st.success(f"Attendance saved to {output_csv_path}!")

# Streamlit UI
def main():
    st.title("Attendance System with Emotion Detection")
    st.text("This application runs from 9:30 AM to 10:00 AM only.")

    # Streamlit components for video input
    run_app = st.checkbox("Run Attendance System")

    # Placeholder for live updates
    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    if run_app:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("No camera input detected!")
                break

            # Process frame
            student_name, emotion = process_frame(frame)

            # Update UI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, caption="Live Camera Feed", use_column_width=True)
            info_placeholder.markdown(f"**Student:** {student_name}  \n**Emotion:** {emotion}")

            # Exit if attendance time is over
            if not is_within_time_window():
                st.warning("Attendance time is over. Saving attendance...")
                break

        cap.release()

        # Save attendance
        save_attendance()

if __name__ == "__main__":
    main()
