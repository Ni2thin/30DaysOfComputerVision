import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from util import classify, set_background  # Ensure these utilities are implemented

# Set custom CSS for background and styling
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        font-family: Arial, sans-serif;
    }
    .main {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Header
st.title("Pneumonia Classification Web App")
st.header("Upload a Chest X-ray Image for Analysis")

# File Upload Section
file = st.file_uploader(
    label="Upload your chest X-ray image",
    type=["jpeg", "jpg", "png"],
    label_visibility="collapsed"
)

# Load Classifier Model
model = load_model('/Users/nitthin/Documents/Computer vision/Pneumonia classification/pneumonia_classifier.h5')

# Load Class Names
with open('/Users/nitthin/Documents/Computer vision/Pneumonia classification/labels.txt', 'r') as f:
    class_names = [line.strip().split(' ')[1] for line in f.readlines()]

# File Processing and Prediction
if file is not None:
    # Display the uploaded image
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    # Classify the image
    class_name, conf_score = classify(image, model, class_names)

    # Convert confidence score to percentage
    conf_percentage = int(conf_score * 100)

    # Display Prediction Results
    st.write(f"### Prediction: {class_name}")
    st.write(f"### Confidence: {conf_percentage}%")
    
    # Display Prediction Confidence as a Progress Bar
    st.progress(conf_percentage)

    # Additional Styling for Confidence Score
    st.markdown(
        f"""
        <div style="font-size: 18px; font-weight: bold; color: {'green' if conf_score > 0.8 else 'orange'};">
            Model is {'highly' if conf_score > 0.8 else 'moderately'} confident in this prediction.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("Please upload a chest X-ray image to begin.")
