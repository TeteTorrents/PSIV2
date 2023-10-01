import os
import cv2
from charSegment_pipeline import segmentChars
from utils import *


def comparar_pred(bb_pred, bb_gt, roi, iou_threshold):

    # Convertim les coord de bb_gt  de format YOLOv8 a (x, y, w, h)
    x, y, w, h = yolov8_to_xywh(bb_gt, roi.shape[1], roi.shape[0])
    adapted_bb_gt = [x, y, w, h]

    # Calculem IoU entre la predicció la bbox real
    iou = calculate_iou(bb_pred, adapted_bb_gt, roi, False)

    # Miren en funció d'un threshold si la bboxes detectada pel nostre sistema és correcte o no
    if iou >= iou_threshold:
        return "Predicció correcta", iou
    else:
        return "Predicció incorrecta", iou


# Directori amb el test
images_dir = './matricules/test_segmentation/images'
labels_dir = './matricules/test_segmentation/labels'
iou_threshold = 0.65

prediccions_correctes = 0
imatges_totals = 0
iou_total = 0.0

# Iterem per totes les imatges i calculem la accur + IoU mitjana
for image_filename in os.listdir(images_dir):
    if image_filename.endswith('.jpg') or image_filename.endswith('.jpeg'):
        image_path = os.path.join(images_dir, image_filename)
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)

        # Llegim la imatge i la passem a gris
        image = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detectem els chars
        bboxes_pred = segmentChars(image_gray, image)
        bboxes_pred = sorted(bboxes_pred, key = lambda x: x[0])

        # Obtenim els bboxes reals (groundtruth)
        bboxes_gt = []
        with open(label_path, 'r') as label_file:
            # Llegim totes les bboxes, linea per linea
            lines = label_file.readlines()
            for line in lines:
                label_data = line.strip().split()
                bboxes_gt.append(label_data[1:])
        
        for bb_gt, bb_pred in zip(bboxes_gt, bboxes_pred):
            detection_result, iou = comparar_pred(bb_pred, bb_gt, image, iou_threshold)
            iou_total += iou
            imatges_totals += 1
            
            if detection_result == "Predicció correcta":
                prediccions_correctes += 1

# Calculate accuracy and average IoU
accuracy = (prediccions_correctes / imatges_totals) * 100 if imatges_totals > 0 else 0
average_iou = iou_total / imatges_totals if imatges_totals > 0 else 0

print(f"Accuracy: {accuracy:.2f}%")
print(f"IoU mitjà: {average_iou:.4f}")