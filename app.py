oimport streamlit as st
import torch
import pickle
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import datetime

# Load the trained model and label encoder
model_path = "emotion_detection_model.pkl"
label_encoder_path = "label_encoder.pkl"

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Set the model to evaluation mode
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Streamlit app title
st.title("Emotion Recognition System")

# Upload an image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_image = transform(image).unsqueeze(0)

    # Predict emotion
    with torch.no_grad():
        outputs = model(input_image)
        _, predicted = torch.max(outputs, 1)
        predicted_label = label_encoder.inverse_transform([predicted.item()])[0]

    st.write(f"Predicted Emotion: **{predicted_label}**")

    # Option to download results as CSV
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result_data = pd.DataFrame({'Timestamp': [timestamp], 'Emotion': [predicted_label]})
    
    csv = result_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='emotion_results.csv',
        mime='text/csv'
    )

