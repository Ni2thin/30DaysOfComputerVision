import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Prepare data
input_dir = '/Users/nitthin/Documents/Computer vision/image detection /clf-data'
categories = ['empty', 'not_empty']

data = []
labels = []

for category_idx, category in enumerate(categories):
    category_path = os.path.join(input_dir, category)
    for file in os.listdir(category_path):
        img_path = os.path.join(category_path, file)
        try:
            img = imread(img_path)
            img_resized = resize(img, (15, 15), anti_aliasing=True)
            data.append(img_resized.flatten())
            labels.append(category_idx)
        except Exception as e:
            print(f"Error processing file {img_path}: {e}")

data = np.asarray(data)
labels = np.asarray(labels)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# Train classifier
classifier = SVC()

parameters = {'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}

grid_search = GridSearchCV(
    estimator=classifier, param_grid=parameters, cv=5, scoring='accuracy'
)

grid_search.fit(x_train, y_train)

# Test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print(f'{score * 100:.2f}% of samples were correctly classified.')

# Save the model
with open('./model.p', 'wb') as model_file:
    pickle.dump(best_estimator, model_file)
