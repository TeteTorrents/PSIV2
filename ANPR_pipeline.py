import os
import cv2

from individual_char_recog import svm_recognizer, nn_recognizer
from plateDetection_pipeline import detect_plate

def anpr_pipeline(image_path, mode = 'Yolo'):

    # Load original image
    image_o = cv2.imread(image_path)
    image_o = cv2.resize(image_o, (1000, 800))

    # Detect licence plate
    x_roi, y_roi, w_roi, h_roi, image = detect_plate(image_path)
    roi = image[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

    if mode == 'Yolo':
        recog_result = ...
    elif mode == 'SVM':
        recog_result = svm_recognizer(roi)
    elif mode == 'Xarxa':
        recog_result = nn_recognizer(roi)

    # Print the text
    text = ''.join(recog_result)
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

#path = 'fotos_profe/test3.jpg'
#anpr_pipeline(path, "Xarxa")
#print("DONE!")