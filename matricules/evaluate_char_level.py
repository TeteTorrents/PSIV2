import os
import cv2
from charSegment_pipeline import segmentChars
from individual_char_recog import *
from utils import *
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import re
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Directori amb el test
images_dir = './matricules/test_segmentation/images'

# Hard-code labels
labels_aux = ["2765HKR", "7784JGC", "7549LMH", "1296KSV", "2765HKR", "4135FNX", "1074KSN", "8622LW", "2765HKR",
            "5479HHR", "3572CLX", "1842JJN", "6729HH", "7105GV", "9848HSZ", "3235MKB", "7718GZC", "1074KSN",
            "2907HTR", "8622L", "4287KGJ", "1074KSN", "2976LY", "9101FLC", "5865LZT", "7362LWV", "5895KKT"]
ind = [1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30]
labels_dict = cotxe_dict = {f'cotxe{i}': label for i, label in zip(ind, labels_aux)}
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
        easyOcr_predictions.append(list(re.sub(r'[^A-Za-z1-9]', '', easyOCR_recognizer(image))))
        pyt_predictions.append(pytesseract_recognizer(image))

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
    df_cm = pd.DataFrame(confusion_matrices[i], index = [i for i in lletres_cm[i]],
                  columns = [i for i in lletres_cm[i]])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=False, cmap=blueish_palette)
    plt.title(f"Confusion Matrix {metodes[i]}")
    plt.savefig(f"./matricules/heatmaps/hm_{metodes[i]}.png")


pra_char = {}
for i in range(len(metodes)):
    pra_char[metodes[i]] = {}
    for char_index, char in enumerate(lletres_cm[i]):
        
        # Calculem la precisió per cada lletra
        true_positive = confusion_matrices[i][char_index, char_index]
        false_positive = sum(confusion_matrices[i][:, char_index]) - true_positive
        false_negative = sum(confusion_matrices[i][char_index, :]) - true_positive
        precision = true_positive / (true_positive + false_positive)
        
        # Calculem la recall
        recall = true_positive / (true_positive + false_negative)
        
        # Calcula l'accur
        accuracy = confusion_matrices[i][char_index][char_index] / np.sum(confusion_matrices[i][char_index])
        
        pra_char[metodes[i]][char] = [accuracy, precision, recall]

df = pd.DataFrame(pra_char)
df = df.transpose()
print(df)
print("DONE!")