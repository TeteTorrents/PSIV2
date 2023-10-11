import os
import cv2
import pandas as pd
from matricules.char_recognition.individual_char_recog import *
from matricules.detection.plateDetection_pipeline import detect_plate_classic, detect_plate_yolo
import prettytable as pt
from tabulate import tabulate
from dotenv import load_dotenv

# Carreguem variables d'entorn
load_dotenv()

def anpr_pipeline(image_path, recog_mode = 'SVM', detect_mode = 'Yolo', debug = False):

    # Load original image
    image_o = cv2.imread(image_path)
    image_o = cv2.resize(image_o, (1000, 800))

    # Detect licence plate
    if detect_mode == 'Classic':
        x_roi, y_roi, w_roi, h_roi, image_ge, image_norm = detect_plate_classic(image_path, debug)
    elif detect_mode == 'Yolo':
        x_roi, y_roi, w_roi, h_roi, image_ge, image_norm = detect_plate_yolo(image_path, debug)
        x_roi, y_roi, w_roi, h_roi = int(x_roi), int(y_roi), int(w_roi), int(h_roi)
    roi_ge = image_ge[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
    roi_o = image_norm[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

    if roi_o.size == 0:
        return None
    else:
        if recog_mode == 'SVM':
            recog_result = svm_recognizer(roi_ge, roi_o)
        elif recog_mode == 'Xarxa':
            recog_result = nn_recognizer(roi_ge, roi_o)
        elif recog_mode == 'EasyOCR':
            recog_result = easyOCR_recognizer(roi_ge, roi_o)
        elif recog_mode == 'Pytesseract':
            recog_result = pytesseract_recognizer(roi_ge, roi_o)

    text = ''.join(recog_result)
    if debug:
        # Print the text
        print(text)

        # Calculate the relative coords and show ROI
        y, x, _ = image_o.shape
        crop_y_start, crop_x_start = int(1/3 * y), int(1/3 * x)
        x1_o, y1_o, x2_o, y2_o = x_roi + crop_x_start, y_roi + crop_y_start, x_roi+w_roi + crop_x_start, y_roi+h_roi + crop_y_start
        cv2.rectangle(image_o, (x1_o, y1_o), (x2_o, y2_o), (0, 255, 0), 2)
        cv2.imshow('Image with Bounding Boxes', image_o)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return text