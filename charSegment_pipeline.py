import cv2

def segmentChars(roi):

    # Passem la imatge a gris
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Fem "Black hat"
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    closed_roi = cv2.morphologyEx(roi_gray, cv2.MORPH_CLOSE, kernel, iterations = 5)
    blackHat_roi = closed_roi - roi_gray

    # Binaritzem la imatge amb thresholding
    _, binary_roi = cv2.threshold(blackHat_roi, 120, 255, cv2.THRESH_BINARY)

    # Trobem els elements a segmentar
    totalLabels_roi, labelsInfo_roi, stats_roi, centroids_roo = cv2.connectedComponentsWithStats(binary_roi, 8, cv2.CV_32S)

    letters_bboxes = []
    for label in range(1, totalLabels_roi):
        x, y, w, h, _ = stats_roi[label]
        if h > 15 and w > 15 and (w < roi.shape[1]/3):
            cv2.rectangle(roi, (x-2, y-5), (x-2 + w+5, y-5 + h+10), (0, 255, 0), 2)
            letters_bboxes.append((x-2,y-5,w+2,h+10))
    
    cv2.imshow('Image with Bounding Boxes', roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return letters_bboxes

path_org = "fotos_profe/PXL_20210921_095212294.jpg"
img_org = cv2.imread(path_org)
image = cv2.resize(img_org, (1000, 800))
x_roi, y_roi, w_roi, h_roi = (147, 426, 190, 82)
roi = image[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

cv2.imshow('Image with Bounding Boxes', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

segmentChars(roi)
