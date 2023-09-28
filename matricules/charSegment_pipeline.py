import cv2
from feature_extractor import calculate_pixel_density

def segmentChars(roi, roi_o, debug = False):

    # Fem "Black hat"
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed_roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations = 5)
    blackHat_roi = closed_roi - roi

    # Opening al blackhat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    open_roi = cv2.morphologyEx(blackHat_roi, cv2.MORPH_OPEN, kernel, iterations = 1)

    # Closing al opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    closed_roi2 = cv2.morphologyEx(open_roi, cv2.MORPH_OPEN, kernel, iterations = 1)

    # Binaritzem la imatge amb thresholding
    ret_val, otsu_binary = cv2.threshold(closed_roi2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Trobem els elements a segmentar
    totalLabels_roi, labelsInfo_roi, stats_roi, centroids_roo = cv2.connectedComponentsWithStats(otsu_binary, 4, cv2.CV_32S)

    idx_letters = []
    for label in range(1, totalLabels_roi):
        x, y, w, h, _ = stats_roi[label]
        if h > 15 and w > 5 and (w < roi.shape[1]/3):
            idx_letters.append((label, calculate_pixel_density(roi_o[y:y+h, x:x+w])))
    idx_letters_sorted = sorted(idx_letters, key=lambda x: x[1], reverse = True)
    letters_bboxes = []
    for i,_ in idx_letters_sorted[:7]:
        x, y, w, h, _ = stats_roi[i]
        #cv2.rectangle(roi_o, (x-2, y-5), (x-2 + w+5, y-5 + h+10), (255, 255, 0), 2)
        letters_bboxes.append((max(0, x-2),max(0, y-5),w+2,h+10))
    
    if debug:
        cv2.imshow('Image with Bounding Boxes', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return letters_bboxes

"""
path_org = "fotos_profe/PXL_20210921_095212294.jpg"
img_org = cv2.imread(path_org)
image = cv2.resize(img_org, (1000, 800))
x_roi, y_roi, w_roi, h_roi = (147, 426, 190, 82)
roi = image[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

cv2.imshow('Image with Bounding Boxes', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

segmentChars(roi)
"""