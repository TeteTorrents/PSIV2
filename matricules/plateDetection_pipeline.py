import cv2
import numpy as np
import imutils

def detect_plate(image_path):

    # Llegim la imatge
    image_o = cv2.imread(image_path)
    image_o = cv2.resize(image_o, (1000, 800))
    y,x,_ = image_o.shape
    image = image_o[int(1/3*y):, int(1/3*x):]

    # Passem la imatge a escala de grisos (?)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Fem equalització del histograma
    gray_equalized = cv2.equalizeHist(gray)

    #Fem otsu
    ret_val, otsu_binary = cv2.threshold(gray_equalized, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

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
    hist1 = np.histogram(open_image.flatten())


    # Binaritzem la imatge mitjançant la operació de thresholding (definim un threshold de 80)
    _, binary_image = cv2.threshold(open_image, 85, 255, cv2.THRESH_BINARY)

    # Erosionem i dilatem per eliminar elements/soroll de la imatge binaritzada
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    rem_elem_image = cv2.erode(binary_image, kernel, iterations=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded_image = cv2.dilate(rem_elem_image, kernel, iterations=3)

    # De la imatge resultant obtenim el elements (per segmentar la licence plate) amb l'algoritme de Connected Component Labeling
    totalLabels, labelsInfo, stats, centroids = cv2.connectedComponentsWithStats(eroded_image, 8, cv2.CV_32S)

    # Iterem per tots els elemnts trobats i ens quedem amb la bbox de la licence plate
    # AIXÒ S'HA D'ARREGLAR PERQUÈ HO FACI BÉ!!!!!! (LA PART DE LA CONDICIÓ PRINCIPALMENT)

    cnt_rat = 0
    for label in range(1, totalLabels):
        x, y, w, h, _ = stats[label]
        if 2.5 <= w/h <= 6:
            cnt_rat += 1

    for label in range(1, totalLabels):
        x, y, w, h, _ = stats[label]
        if cnt_rat > 1:
            if 2.5 <= w/h <= 6 and w < image.shape[0]/1.5 and h < image.shape[1]/4 and y > image.shape[1]/4:
                # Dibuixem un bbox a la imatge i guardem la info de les coordenades
                cv2.rectangle(image, (x-20, y-10), (x - 20 + w + 40, y - 10 + h + 15), (255, 255, 0), 2)  # (0, 255, 0) is the color of the rectangle (green), 2 is the line thickness
                x_roi, y_roi, w_roi, h_roi = x-20, y - 10, w + 40, h + 15
        else:
            if 2.5 <= w/h <= 6:
                cv2.rectangle(image, (x-20, y-10), (x - 20 + w + 40, y - 10 + h + 15), (255, 255, 0), 2)  # (0, 255, 0) is the color of the rectangle (green), 2 is the line thickness
                x_roi, y_roi, w_roi, h_roi = x-20, y - 10, w + 40, h + 15

    #cv2.imshow('Image with Bounding Boxes', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return x_roi, y_roi, w_roi, h_roi, image