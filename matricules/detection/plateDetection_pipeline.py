import sys
sys.path.insert(0, r'C:\Users\adars\OneDrive\Escritorio\Uni\PSIV2\PSIV2')

import os
import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

def detect_plate_classic(image_path, debug = False):

    # Llegim la imatge
    image_o = cv2.imread(image_path)
    image_o = cv2.resize(image_o, (1000, 800))
    y,x,_ = image_o.shape
    image = image_o[int(1/3*y):, int(1/3*x):]

    # Passem la imatge a escala de grisos
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Fem equalització del histograma
    gray_equalized = cv2.equalizeHist(gray)

    # Fem un closing -> dilatar + erosionar
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed_image = cv2.morphologyEx(gray_equalized, cv2.MORPH_CLOSE, kernel, iterations = 2)

    # Fem la operació morfològica de "Black Hat" ->  difference between the closing and the given image
    blackHat_image = closed_image - gray_equalized

    # Fem closing a la imatge resultant del "Black Hat"
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 2))
    closed_image2 = cv2.morphologyEx(blackHat_image, cv2.MORPH_CLOSE, kernel, iterations = 2)

    # Fem opening -> erosionar + dilatar
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 25))
    open_image = cv2.morphologyEx(closed_image2, cv2.MORPH_OPEN, kernel, iterations = 1)

    # Binaritzem la imatge mitjançant la operació de thresholding (definim un threshold de 80)
    _, binary_image = cv2.threshold(open_image, 85, 255, cv2.THRESH_BINARY)

    # Erosionem i dilatem per eliminar elements/soroll de la imatge binaritzada
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    rem_elem_image = cv2.erode(binary_image, kernel, iterations=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded_image = cv2.dilate(rem_elem_image, kernel, iterations=3)

    # De la imatge resultant obtenim el elements (per segmentar la licence plate) amb l'algoritme de Connected Component Labeling
    totalLabels, _, stats, _ = cv2.connectedComponentsWithStats(eroded_image, 8, cv2.CV_32S)

    # Iterem per tots els elemnts trobats i ens quedem amb la bbox de la licence plate
    cnt_rat = 0
    for label in range(1, totalLabels):
        x, y, w, h, _ = stats[label]
        if 2.5 <= w/h <= 6:
            cnt_rat += 1

    bboxes = []
    for label in range(1, totalLabels):
        x, y, w, h, _ = stats[label]
        if cnt_rat > 1:
            if 2.5 <= w/h <= 6 and w < image.shape[0]/1.5 and h < image.shape[1]/4 and y > image.shape[1]/5.5 and x > 10 and y < image.shape[1]/1.5:
                x_roi, y_roi, w_roi, h_roi = max(0, x-20), max(0, y - 10), w + 40, h + 15
                bboxes.append([x_roi, y_roi, w_roi, h_roi])
        else:
            if 2.5 <= w/h <= 6:
                x_roi, y_roi, w_roi, h_roi = max(0, x-20), max(0, y - 10), w + 40, h + 15
                bboxes.append([x_roi, y_roi, w_roi, h_roi])

    if not bboxes:
        x_roi, y_roi, w_roi, h_roi = 0, 0, 0, 0
    elif len(bboxes) == 1:
        x_roi, y_roi, w_roi, h_roi = bboxes[0]
    else:
        sorted_bboxes = sorted(bboxes, key=lambda bbox: bbox[2] * bbox[3])
        x_roi, y_roi, w_roi, h_roi = sorted_bboxes[0]

    if debug:
        cv2.rectangle(image, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (255, 255, 0), 2)

        cv2.imshow('Image with Bounding Boxes', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return x_roi, y_roi, w_roi, h_roi, gray_equalized, image

def detect_plate_yolo(image_path, debug = False):
    # Llegim la imatge
    image_o = cv2.imread(image_path)
    image_o = cv2.resize(image_o, (1000, 800))
    y,x,_ = image_o.shape
    image = image_o[int(1/3*y):, int(1/3*x):]

    # Passem la imatge a escala de grisos
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Fem equalització del histograma
    gray_equalized = cv2.equalizeHist(gray)

    # Detectem la placa amb YOLO
    model = YOLO(r'C:\Users\adars\OneDrive\Escritorio\Uni\PSIV2\PSIV2\matricules\models\Yolov8\models\license_plate_detector.pt')
    results = model(image)
    x1,y1,x2,y2 = results[0].boxes.xyxy[0]
    x,y,w,h = x1,y1,x2-x1,y2-y1

    if debug:
        cv2.rectangle(image, (int(x), int(y)), (int(x)+int(w), int(y)+int(h)), (255, 255, 0), 2)  

        cv2.imshow('Image with Bounding Boxes', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()        

    return x, y, w, h, gray_equalized, image