import os
import cv2
import numpy as np
from utils import get_face_landmarks

# Set threading limits
import os
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
import cv2
cv2.setNumThreads(1)

# Path to the data directory
data_dir = '/Users/nitthin/Documents/Computer vision/emotion detection/data'

# Function to process a batch of images
def process_batch(image_paths, emotion_indx, emotion_name):
    batch_output = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue

        face_landmarks = get_face_landmarks(image)
        print(f"Processing {emotion_name}/{os.path.basename(image_path)}: {len(face_landmarks)} landmarks")

        if len(face_landmarks) == 1404:
            face_landmarks.append(int(emotion_indx))
            batch_output.append(face_landmarks)
    return batch_output

# Process each emotion folder
for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    emotion_path = os.path.join(data_dir, emotion)

    if not os.path.isdir(emotion_path):
        continue

    # Get all image paths in the folder
    image_paths = [
        os.path.join(emotion_path, img) for img in os.listdir(emotion_path)
        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]

    batch_size = 10  # Number of images to process at once
    output = []

    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        output.extend(process_batch(batch, emotion_indx, emotion))

    # Save intermediate results for the current emotion
    intermediate_file = os.path.join(data_dir, f'data_{emotion}.txt')
    np.savetxt(intermediate_file, np.asarray(output))
    print(f"Processed {emotion}, saved to {intermediate_file}")

# Combine all intermediate results into a single file
combined_file = os.path.join(data_dir, 'data.txt')

with open(combined_file, 'w') as outfile:
    for emotion_file in sorted(os.listdir(data_dir)):
        if emotion_file.startswith('data_') and emotion_file.endswith('.txt'):
            emotion_file_path = os.path.join(data_dir, emotion_file)
            with open(emotion_file_path, 'r') as infile:
                outfile.write(infile.read())

print(f"All data combined into {combined_file}")
