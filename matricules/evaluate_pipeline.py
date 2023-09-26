from ANPR_pipeline import anpr_pipeline

from nltk.metrics import edit_distance
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import os

def evaluate_anpr_pipeline(img_dir, mode = 'Yolo'):
    
    # List to store recognized values and true labels
    true_labels = []
    recognized_labels = []

    # Iterem per tots els arxius del dir
    for filename in os.listdir(img_dir):
        if filename.endswith('.jpeg'):
            image_path = os.path.join(img_dir, filename)
            true_label = os.path.splitext(filename)[0]

            # Utilitzem pla ANPR pipeline per trobar el text que se'ns detecta
            recognized_label = anpr_pipeline(image_path, mode)

            # Afegim els elements a la llista
            true_labels.append(true_label)
            recognized_labels.append(recognized_label)

    ### Calculem les mètriques

    # Character-Level Accuracy (CLA)
    total_characters = sum(len(label) for label in true_labels)
    correct_characters = sum(edit_distance(true, recognized) for true, recognized in zip(true_labels, recognized_labels))
    cla = 1 - (correct_characters / total_characters)

    # Word-Level Accuracy (WLA)
    correct_words = sum(1 for true, recognized in zip(true_labels, recognized_labels) if true == recognized)
    wla = correct_words / len(true_labels)

    # Edit Distance
    edit_distances = [edit_distance(true, recognized) for true, recognized in zip(true_labels, recognized_labels)]

    # Confusion Matrix
    all_characters = set(''.join(true_labels + recognized_labels))
    #cm = confusion_matrix(''.join(true_labels), ''.join(recognized_labels), labels=list(all_characters))

    # F1 Score, recall i precision
    f1 = f1_score(true_labels, recognized_labels, average='weighted')
    recall = recall_score(true_labels, recognized_labels, average='weighted')
    precision = precision_score(true_labels, recognized_labels, average='weighted')

    return cla, wla, edit_distances, "cm", f1, recall, precision


evaluate_anpr_pipeline('./img_test_final', 'Xarxa')