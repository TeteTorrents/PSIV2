import os
import cv2

# Convertir les coordenades Yolo de roboflow a xywh
def yolov8_to_xywh(yolov8_bbox, image_width, image_height):
    x_center, y_center, bbox_width, bbox_height = map(float, yolov8_bbox)
    x = (x_center - bbox_width / 2.0) * image_width
    y = (y_center - bbox_height / 2.0) * image_height
    w = bbox_width * image_width
    h = bbox_height * image_height
    return x, y, w, h

# Funció per adaptar les bboxes que tenim a la imatge canviada de tamany
def adaptar_bbox_coords(bboxes, original_image_size, new_image_size):
    x_ratio = new_image_size[0] / original_image_size[0]
    y_ratio = new_image_size[1] / original_image_size[1]

    adapted_bboxes = []
    for bbox in bboxes:
        x, y, w, h = bbox
        x *= x_ratio
        y *= y_ratio
        w *= x_ratio
        h *= y_ratio
        adapted_bboxes.append([x, y, w, h])
    return adapted_bboxes

# Calcular IoU
def calculate_iou(bbox1, bbox2, image_o, plate_mode = True, debug = False):

    # Adaptem les cooordenades predicted per nosaltres (perquè son en funció de la imatge retallada)
    y, x, _ = image_o.shape
    if plate_mode:
        crop_y_start, crop_x_start = int(1/3 * y), int(1/3 * x)
        x1_o, y1_o, x2_o, y2_o = bbox1[0] + crop_x_start, bbox1[1] + crop_y_start, bbox1[0]+bbox1[2] + crop_x_start, bbox1[1]+bbox1[3] + crop_y_start
        bbox1 = [x1_o, y1_o, x2_o-x1_o, y2_o-y1_o]

    if debug:
        cv2.rectangle(image_o, (bbox1[0], bbox1[1]), (bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]), (255, 255, 0), 2)
        cv2.rectangle(image_o, (int(bbox2[0]), int(bbox2[1])), (int(bbox2[0]) + int(bbox2[2]), int(bbox2[1]) + int(bbox2[3])), (255, 0, 0), 2)  

        cv2.imshow('Image with Bounding Boxes', image_o)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Calculem el "rectangle d'intersecció entre les 2 bboxes"
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    # Calculate l'area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate la area de les dos bboxes, per així després clauclar la IoU
    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]

    # Calculem la IoU
    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)
    return iou