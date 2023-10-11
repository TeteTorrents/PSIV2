import sys
sys.path.insert(0, r'C:\Users\adars\OneDrive\Escritorio\Uni\PSIV2\PSIV2')

import os
import cv2
from matricules.char_segmentation.charSegment_pipeline import segmentChars
from matricules.char_recognition.individual_char_recog import *
from matricules.utils.utils import *
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import re
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Directori amb el test
images_dir = 'C:/Users/adars/OneDrive/Escritorio/Uni/PSIV2/PSIV2/matricules/fotos/test_segmentation/images_yolo'

# Hard-code labels
labels_aux = ["2765HKR", "7784JGC", "7549LMH", "1296KSV", "2765HKR", "4135FNX", "1074KSN", "8622LW", "2765HKR",
            "5479HHR", "3572CLX", "3572CLX", "7084JKZ", "1842JJN", "6729HHT", "7105GVT", "9848HSZ", "3235MKB", "7718GZC", "1074KSN",
            "2907HTR", "8622LKW", "4287KGJ", "1074KSN", "2976LYC", "9101FLC", "5865LZT", "9057HST", "7362LWV", "5895KKT"]
labels_dict = cotxe_dict = {f'cotxe{i+1}': label for i, label in enumerate(labels_aux)}
labels = []

# Mètriques
accuracies = []
confusion_matrices = []
lletres_cm = []

# Predictions
svm_predictions = []
nn_predictions = []
easyOcr_predictions = []
pyt_predictions = []

# Iterem per totes les imatges i calculem la accur + IoU mitjana
for image_filename in os.listdir(images_dir):
    if image_filename.endswith('.jpg') or image_filename.endswith('.jpeg'):
        image_path = os.path.join(images_dir, image_filename)
        labels.append(labels_dict[image_filename.split('_')[0]])
        # Llegim la imatge i la passem a gris
        image = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Obtenim la predicció per cada model
        svm_predictions.append(svm_recognizer(image_gray, image))
        nn_predictions.append(nn_recognizer(image_gray, image))
        easyOcr_predictions.append(easyOCR_recognizer(image_gray, image))
        pyt_predictions.append(pytesseract_recognizer(image_gray, image))

# Calculem les mètriques
for i, predictions in enumerate([svm_predictions, nn_predictions, easyOcr_predictions, pyt_predictions]):
    # Inicialitzem helper vars
    total_characters = 0
    total_correct = 0
    true_labels = []
    pred_labels = []

    for j in range(len(labels)):
        true_chars = list(labels[j])
        pred_chars = list(predictions[j])
        
        if len(true_chars) > len(pred_chars):
            pred_chars = pred_chars + true_chars[len(pred_chars):]
        else:
            true_chars = true_chars + pred_chars[len(true_chars):]

        total_characters += len(true_chars)
        total_correct += sum([1 if true_chars[k] == pred_chars[k] else 0 for k in range(len(true_chars))])
        
        # Ens guardem la label real i la predita per poder fer la conf matrix
        true_labels.extend(true_chars)
        pred_labels.extend(pred_chars)

    # Calculem mètriques a nivell de caràcter
    accuracy = total_correct / total_characters

    # Fem la matriu de confusió
    confusion = confusion_matrix(true_labels, pred_labels)
    
    # Ens guardem les mètriques
    accuracies.append(accuracy)
    confusion_matrices.append(confusion)
    lletres_cm.append("".join(sorted(list(set(true_labels + pred_labels)))))

metodes = ["SVM", "Xarxa", "easyOCR", "Pytesseract"]
blueish_palette = sn.color_palette("Blues")
for i in range(len(metodes)):
    print(f"Metrics for {metodes[i]}:")
    print(f"Accuracy: {accuracies[i]:.2f}")
    print("Confusion Matrix:\n")
    try:
        df_cm = pd.DataFrame(confusion_matrices[i], index = [i for i in lletres_cm[i]],
                  columns = [i for i in lletres_cm[i]])
    except:
        df_cm = pd.DataFrame(confusion_matrices[i], index = [i for i in lletres_cm[i].replace('_', '')],
                  columns = [i for i in lletres_cm[i].replace('_', '')])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=False, cmap=blueish_palette)
    plt.title(f"Confusion Matrix {metodes[i]}")
    plt.savefig(f"C:/Users/adars/OneDrive/Escritorio/Uni/PSIV2/PSIV2/matricules/heatmaps/hm_{metodes[i]}.png")
