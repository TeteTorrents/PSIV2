import os
import cv2
import pandas as pd
from individual_char_recog import *
from plateDetection_pipeline import detect_plate
import prettytable as pt
from tabulate import tabulate

def anpr_pipeline(image_path, mode = 'Yolo'):

    # Load original image
    image_o = cv2.imread(image_path)
    image_o = cv2.resize(image_o, (1000, 800))

    # Detect licence plate
    x_roi, y_roi, w_roi, h_roi, image_ge, image_norm = detect_plate(image_path)
    roi_ge = image_ge[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
    roi_o = image_norm[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

    if mode == 'Yolo':
        recog_result = ...
    elif mode == 'SVM':
        recog_result = svm_recognizer(roi_ge, roi_o)
    elif mode == 'Xarxa':
        recog_result = nn_recognizer(roi_ge, roi_o)
    elif mode == 'EasyOCR':
        recog_result = easyOCR_recognizer(roi_o)
    elif mode == 'Pytesseract':
        recog_result = pytesseract_recognizer(roi_o)

    # Print the text
    text = ''.join(recog_result)
    print(text)

    # Calculate the relative coords and show ROI
    y, x, _ = image_o.shape
    crop_y_start, crop_x_start = int(1/3 * y), int(1/3 * x)
    x1_o, y1_o, x2_o, y2_o = x_roi + crop_x_start, y_roi + crop_y_start, x_roi+w_roi + crop_x_start, y_roi+h_roi + crop_y_start
    #cv2.rectangle(image_o, (x1_o, y1_o), (x2_o, y2_o), (0, 255, 0), 2)
    #cv2.imshow('Image with Bounding Boxes', image_o)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return text

df = pd.DataFrame(columns=['Model', 'foto', 'Plate'])
types = ['SVM', 'Xarxa','EasyOCR'] #, 'Pytesseract'], 'Yolo']

fotos = [f"cotxe{i}.jpg" for i in range(1, 10)]
for f in fotos:
    # print(t)
    path = os.path.join('./fotos', f)
    for t in types:
        plate = anpr_pipeline(path, t).replace(' ', '')
        df.loc[len(df)] = [t, f, plate]
        print(f"DONE! {t} - {f}")
    # plate = anpr_pipeline(path, t).replace(' ', '')
    # print(plate)
    # df.loc[len(df)] = [t, plate]
    # print(f"DONE! {t}")

print(tabulate(df, headers='keys', tablefmt='psql'))

#path = 'fotos_profe/im8.jpeg'
#anpr_pipeline(path, "SVM")
#print("DONE!")