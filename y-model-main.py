# y-model-main.py

import os
import pickle
import numpy as np
from dotenv import load_dotenv

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load environment variables
load_dotenv()

# Get dataset path from .env file
input_dir = os.getenv("INPUT_DIR")

# Define categories
categories = ['Empty', 'Not Empty']

# Prepare data
data = []
labels = []

print("Loading data and processing images...")
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15)) # Resize image to 15x15 pixels
        data.append(img.flatten())  # Flatten the image to a 1D array
        labels.append(category_idx) # Assign categories index as label

data = np.asarray(data)
labels = np.asarray(labels)

# Split dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

print("Data loading complete!")

# Train classifier
print("Training SVM classifier...")
classifier = SVC()

# Hyperparameter tuning using Grid Search
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# Test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open('./y-model.p', 'wb'))