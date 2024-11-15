import streamlit as st
import pandas as pd
import pickle
import torch
from torchvision import transforms
from PIL import Image
import datetime

# Load model and label encoder
model_path = 'emotion_detection_model.pkl'
label_encoder_path = 'label_encoder.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Function to predict emotion
def predict_emotion(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    emotion = label_encoder.inverse_transform([predicted.item()])[0]
    return emotion

# Attendance data
attendance = []

# Streamlit UI
st.title("Class Attendance and Emotion Detection")
st.write("This app detects students' emotions and marks attendance between 9:30 AM and 10:00 AM.")

# Check if the current time is within the allowed time
now = datetime.datetime.now()
start_time = datetime.time(9, 30)
end_time = datetime.time(10, 0)

if start_time <= now.time() <= end_time:
    uploaded_file = st.file_uploader("Upload a photo of the class:", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Predict emotion
        emotion = predict_emotion(image)
        st.write(f"Detected Emotion: {emotion}")

        # Mark attendance
        name = st.text_input("Enter the student's name:")
        if st.button("Mark Attendance"):
            if name:
                timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
                attendance.append({"Name": name, "Emotion": emotion, "Timestamp": timestamp})
                st.success(f"Attendance marked for {name}!")
            else:
                st.error("Please enter the student's name.")
else:
    st.warning("The attendance system is only active between 9:30 AM and 10:00 AM.")

# Display attendance
if st.button("View Attendance"):
    if attendance:
        attendance_df = pd.DataFrame(attendance)
        st.dataframe(attendance_df)
    else:
        st.warning("No attendance records yet.")

# Download attendance CSV
if attendance:
    attendance_df = pd.DataFrame(attendance)
    csv = attendance_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Attendance CSV",
        data=csv,
        file_name='attendance.csv',
        mime='text/csv'
    )
