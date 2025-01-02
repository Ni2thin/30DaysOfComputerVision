import os
import pickle
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize Img2Vec instance
img2vec = Img2Vec()

# Define dataset directories
base_dir = os.path.abspath('./data/weather_dataset')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# Prepare data
data = {}

for j, dir_ in enumerate([train_dir, val_dir]):
    features = []
    labels = []

    # Iterate over categories
    for category in os.listdir(dir_):
        category_path = os.path.join(dir_, category)

        if not os.path.isdir(category_path):
            continue  # Skip if it's not a directory

        # Iterate over images in category
        for img_path in os.listdir(category_path):
            img_full_path = os.path.join(category_path, img_path)

            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue  # Skip non-image files

            try:
                with Image.open(img_full_path) as img:
                    img = img.convert("RGB")  # Ensure compatibility with all images
                    img_features = img2vec.get_vec(img)
                    features.append(img_features)
                    labels.append(category)
            except Exception as e:
                print(f"Error processing {img_full_path}: {e}")

    # Assign features and labels
    dataset_key = ['training_data', 'validation_data'][j]
    label_key = ['training_labels', 'validation_labels'][j]
    data[dataset_key] = features
    data[label_key] = labels

# Train model
model = RandomForestClassifier(random_state=0)
model.fit(data['training_data'], data['training_labels'])

# Test performance
y_pred = model.predict(data['validation_data'])
score = accuracy_score(data['validation_labels'], y_pred)
print(f"Validation Accuracy: {score:.2f}")

# Save the model
model_path = os.path.abspath('./model.p')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {model_path}")
