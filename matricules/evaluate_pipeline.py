from ANPR_pipeline import anpr_pipeline
import re
from nltk.metrics import edit_distance
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

labels_aux = ["2765HKR", "7784JGC", "7549LMH", "1296KSV", "2765HKR", "4135FNX", "1074KSN", "8622LW", "2765HKR",
              "5479HHR", "3572CLX", "3572CLX", "7084JKZ", "1842JJN", "6729HH", "7105GV", "9848HSZ", "3235MKB",
              "7718GZC", "1074KSN", "2907HTR", "8622L", "4287KGJ", "1074KSN", "2976LY", "9101FLC", "5865LZT",
              "9057HST", "7362LWV", "5895KKT"]
ind = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
labels_dict = cotxe_dict = {f'cotxe{i}': label for i, label in zip(ind, labels_aux)}

def evaluate_anpr_pipeline(img_dir, mode = 'Yolo'):
    
    true_labels = []
    recognized_labels = []

    # Iterem per tots els arxius del dir
    for filename in os.listdir(img_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(img_dir, filename)
            true_label = labels_dict[filename.split('_')[0]]

            # Utilitzem pla ANPR pipeline per trobar el text que se'ns detecta
            recognized_label = anpr_pipeline(image_path, mode)

            # Si el recognized label és None
            if recognized_label is None:
                recognized_labels.append(["ñ"]*len(true_label))
            else:
                # Afegim els elements a la llista
                recognized_labels.append(re.sub(r'[^A-Za-z1-9]', '', recognized_label))
            true_labels.append(true_label)

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

    return cla, wla, edit_distances

cla, wla, edit_distances = evaluate_anpr_pipeline(os.getenv("evaluation_directory"), 'EasyOCR')
print("DONE")