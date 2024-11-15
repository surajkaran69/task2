import streamlit as st
import pickle
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import os
import pandas as pd
import io

# Custom emotion detection model class
class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(EmotionRecognitionModel, self).__init__()
        self.features = models.resnet18(pretrained=False)  # Custom training, no pretraining
        self.features.fc = nn.Linear(self.features.fc.in_features, num_classes)

    def forward(self, x):
        return self.features(x)

# Load model and label encoder
model_path = "emotion_detection_model.pkl"
label_encoder_path = "label_encoding.pkl"

# Load the model and label encoder using pickle
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Set the model to evaluation mode and move it to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Define transformation for input images
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Time check function (Working hours 9:30 AM to 10:00 AM)
def is_within_working_time():
    current_time = datetime.now().time()
    start_time = datetime.strptime("09:30:00", "%H:%M:%S").time()
    end_time = datetime.strptime("10:00:00", "%H:%M:%S").time()
    return start_time <= current_time <= end_time

# Predict emotion and save results to CSV if within the working time
def predict_emotion(image):
    if not is_within_working_time():
        st.warning("Model can only work between 9:30 AM and 10:00 AM.")
        return

    # Load and preprocess image
    image = Image.open(image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Get model predictions
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
        predicted_emotion = label_encoder.inverse_transform([predicted_class.item()])[0]

    # Get current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Save prediction to CSV
    results = {
        "image_path": str(image),
        "predicted_emotion": predicted_emotion,
        "timestamp": timestamp
    }

    # Check if the CSV file exists
    csv_file = "emotion_predictions.csv"
    if os.path.exists(csv_file):
        # Append to CSV if file exists
        df = pd.read_csv(csv_file)
        df = df.append(results, ignore_index=True)
    else:
        # Create a new CSV file if it doesn't exist
        df = pd.DataFrame([results])
    
    # Save the data to CSV
    df.to_csv(csv_file, index=False)

    st.success(f"Prediction: {predicted_emotion}")
    st.write(f"Timestamp: {timestamp}")
    st.write("Prediction saved to CSV.")

    # Provide download link for CSV
    with open(csv_file, 'rb') as f:
        csv_data = f.read()

    # Streamlit's file download feature
    st.download_button(
        label="Download Predictions CSV",
        data=csv_data,
        file_name=csv_file,
        mime="text/csv"
    )

# Streamlit App Interface
def main():
    st.title("Emotion Detection System")
    
    st.write("Upload an image to predict the emotion.")
    
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Predict and show results
        if st.button("Predict Emotion"):
            predict_emotion(uploaded_file)

if __name__ == "__main__":
    main()
