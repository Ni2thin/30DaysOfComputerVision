import os
import pickle
from img2vec_pytorch import Img2Vec
from PIL import Image
import numpy as np

# Load the pre-trained model
model_path = os.path.abspath('./model.p')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from: {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Please train the model first.")
    exit(1)

# Initialize Img2Vec
img2vec = Img2Vec()

# Image path
image_path = os.path.abspath('./data/weather_dataset/val/cloudy/cloudy4.jpg')

# Check if image exists
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    exit(1)

try:
    # Open and process the image
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # Ensure compatibility with all image types
        features = img2vec.get_vec(img)
except Exception as e:
    print(f"Error processing image {image_path}: {e}")
    exit(1)

# Make a prediction
try:
    pred = model.predict([features])
    probabilities = model.predict_proba([features])  # Get confidence scores
    categories = model.classes_  # Retrieve category labels
    confidence = np.max(probabilities)  # Extract highest confidence score
    predicted_label = categories[np.argmax(probabilities)]  # Get the label

    print(f"Prediction: {predicted_label}")
    print(f"Confidence Score: {confidence:.2f}")
except Exception as e:
    print(f"Error during prediction: {e}")
    exit(1)

# Log results
print(f"Image Path: {image_path}")
print(f"Model Details: {type(model).__name__}")
