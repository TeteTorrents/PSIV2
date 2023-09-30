import os
import cv2
from plateDetection_pipeline import detect_plate
from utils import *

def comparar_pred(image_path, label_path, new_image_size, iou_threshold=0.8):
    # Llegim la imatge
    image = cv2.imread(image_path)
    original_image_size = (image.shape[1], image.shape[0])

    # Fem el resize
    image = cv2.resize(image, new_image_size)

    # Llegim les labels de roboflow
    with open(label_path, 'r') as label_file:
        label_data = label_file.read().strip().split()
        yolov8_bbox = label_data[1:]  # Roboflow posa com a primer element de cada fila la classe (0), no ens cal

    # Convertim el format YOLOv8 a (x, y, w, h)
    x, y, w, h = yolov8_to_xywh(yolov8_bbox, original_image_size[0], original_image_size[1])

    # Adaptem les coords de les bboxes a la imatge canviada de mida/escalada
    adapted_bbox = [x, y, w, h]
    adapted_bboxes = adaptar_bbox_coords([adapted_bbox], original_image_size, new_image_size)

    # Calculem la bbox a partir del nostre sistema
    predicted_bbox = detect_plate(image_path)

    # Calculem IoU entre la predicció la bbox real
    iou = calculate_iou(predicted_bbox, adapted_bboxes[0], image)

    # Miren en funció d'un threshold si la bboxes detectada pel nostre sistema és correcte o no
    if iou >= iou_threshold:
        return "Predicció correcta", iou
    else:
        return "Predicció incorrecta", iou

# Directori amb el test
images_dir = './test/images'
labels_dir = './test/labels'
mida_imatge_resized = (1000, 800)
iou_threshold = 0.8

prediccions_correctes = 0
imatges_totals = 0
iou_total = 0.0

# Iterem per totes les imatges i calculem la accur + IoU mitjana
for image_filename in os.listdir(images_dir):
    if image_filename.endswith('.jpg') or image_filename.endswith('.jpeg'):
        image_path = os.path.join(images_dir, image_filename)
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        
        detection_result, iou = comparar_pred(image_path, label_path, mida_imatge_resized, iou_threshold)
        iou_total += iou
        imatges_totals += 1
        
        if detection_result == "Predicció correcta":
            prediccions_correctes += 1

# Calculate accuracy and average IoU
accuracy = (prediccions_correctes / imatges_totals) * 100 if imatges_totals > 0 else 0
average_iou = iou_total / imatges_totals if imatges_totals > 0 else 0

print(f"Accuracy: {accuracy:.2f}%")
print(f"IoU mitjà: {average_iou:.4f}")
