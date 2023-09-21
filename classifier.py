import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from feature_extractor import *
import os

image_data = []
character_labels = []
image_directory = "img_mat_esp"
for filename in os.listdir(image_directory):
    if filename.endswith(".png"):  
        image_path = os.path.join(image_directory, filename)        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_data.append(image)
        character_labels.append(filename.split(".")[0])

feature_vectors = []  # List of feature vectors
for image in character_labels:
    feature_vector = calculate_feature_vector(image)
    feature_vectors.append(feature_vector)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feature_vectors, character_labels, test_size=0.2, random_state=42)
