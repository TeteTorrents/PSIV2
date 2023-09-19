import cv2
import numpy as np
import imutils

def detect_plate(image_path):

    # Llegim la imatge
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1000, 800))

    # Passem la imatge a escala de grisos (?)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Fem un closing -> dilatar + erosionar
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed_image = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations = 2)

    # Fem la operació morfològica de "Black Hat" ->  difference between the closing and the given image
    blackHat_image = closed_image - gray

    # Fem closing a la imatge resultant del "Black Hat"
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
    closed_image2 = cv2.morphologyEx(blackHat_image, cv2.MORPH_CLOSE, kernel, iterations = 1)

    # Fem opening -> erosionar + dilatar
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 35))
    open_image = cv2.morphologyEx(closed_image2, cv2.MORPH_OPEN, kernel, iterations = 1)

    # Binaritzem la imatge mitjançant la operació de thresholding (definim un threshold de 80)
    _, binary_image = cv2.threshold(open_image, 70, 255, cv2.THRESH_BINARY)

    # Erosionem i dilatem per eliminar elements/soroll de la imatge binaritzada
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    rem_elem_image = cv2.erode(binary_image, kernel, iterations=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded_image = cv2.dilate(rem_elem_image, kernel, iterations=3)

    # De la imatge resultant obtenim el elements (per segmentar la licence plate) amb l'algoritme de Connected Component Labeling
    totalLabels, labelsInfo, stats, centroids = cv2.connectedComponentsWithStats(eroded_image, 8, cv2.CV_32S)

    # Iterem per tots els elemnts trobats i ens quedem amb la bbox de la licence plate
    # AIXÒ S'HA D'ARREGLAR PERQUÈ HO FACI BÉ!!!!!! (LA PART DE LA CONDICIÓ PRINCIPALMENT)
    for label in range(1, totalLabels):
        x, y, w, h, _ = stats[label]
        if 2 < w/h < 5:
            # Dibuixem un bbox a la imatge i guardem la info de les coordenades
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)  # (0, 255, 0) is the color of the rectangle (green), 2 is the line thickness
            x_roi, y_roi, w_roi, h_roi = x, y, w, h
    
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return x_roi, y_roi, w_roi, h_roi 


print(detect_plate("fotos_profe/PXL_20210921_095212294.jpg"))