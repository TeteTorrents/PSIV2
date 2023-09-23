import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from feature_extractor import *
import os
import pickle

image_data = []
character_labels = []
image_directory = "img_mat_esp"
for filename in os.listdir(image_directory):
    if filename.endswith(".png"):  
        image_path = os.path.join(image_directory, filename)        
        image = cv2.imread(image_path)
        image_data.append(image)
        character_labels.append(filename.split(".")[0])

feature_vectors = []
for image in image_data:
    feature_vector = calculate_feature_vector(image)
    feature_vectors.append(feature_vector)

# Dividim numeros i lletres
num_labels = character_labels[:10]
letter_labels = character_labels[10:]
labels = [1 if label in num_labels else 0 for label in character_labels]

# Definim SVM per numeros
svm_classifier_numbers = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier_numbers.fit([feature_vectors[i] for i in range(len(labels)) if labels[i] == 1], num_labels)

# Definim SVM per lletres
svm_classifier_letters = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier_letters.fit([feature_vectors[i] for i in range(len(labels)) if labels[i] == 0], letter_labels)

# Evaluem els dos models
y_pred_numbers = svm_classifier_numbers.predict([feature_vectors[i] for i in range(len(labels)) if labels[i] == 1])
y_pred_letters = svm_classifier_letters.predict([feature_vectors[i] for i in range(len(labels)) if labels[i] == 0])

# Calculem la accur pels dos
accuracy_numbers = accuracy_score(num_labels, y_pred_numbers)
accuracy_letters = accuracy_score(letter_labels, y_pred_letters)

"""
# Calculem precision, recall, F-1
precision_numbers = precision_score(num_labels, y_pred_numbers)
recall_numbers = recall_score(num_labels, y_pred_numbers)
f1_numbers = f1_score(num_labels, y_pred_numbers)

precision_letters = precision_score(letter_labels, y_pred_letters)
recall_letters = recall_score(letter_labels, y_pred_letters)
f1_letters = f1_score(letter_labels, y_pred_letters)

print("Metrics for Numbers Classifier:")
print("Accuracy:", accuracy_numbers)
print("Precision:", precision_numbers)
print("Recall:", recall_numbers)
print("F1 Score:", f1_numbers)

print("\nMetrics for Letters Classifier:")
print("Accuracy:", accuracy_letters)
print("Precision:", precision_letters)
print("Recall:", recall_letters)
print("F1 Score:", f1_letters)

# Classification report
print("\nClassification Report for Numbers Classifier:")
print(classification_report(num_labels, y_pred_numbers))

print("\nClassification Report for Letters Classifier:")
print(classification_report(letter_labels, y_pred_letters))
"""
with open('svm_model_numbers.pkl', 'wb') as model_file:
    pickle.dump(svm_classifier_numbers, model_file)

with open('svm_model_letters.pkl', 'wb') as model_file:
    pickle.dump(svm_classifier_letters, model_file)