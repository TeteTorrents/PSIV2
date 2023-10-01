import os
import cv2
from charSegment_pipeline import segmentChars
from individual_char_recog import *
from utils import *


# Directori amb el test
images_dir = './matricules/test_segmentation/images'

# Iterem per totes les imatges i calculem la accur + IoU mitjana
for image_filename in os.listdir(images_dir):
    if image_filename.endswith('.jpg') or image_filename.endswith('.jpeg'):
        image_path = os.path.join(images_dir, image_filename)

        # Llegim la imatge i la passem a gris
        image = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detectem els chars
        bboxes_pred = segmentChars(image_gray, image)
        bboxes_pred = sorted(bboxes_pred, key = lambda x: x[0])

        # Hard-code labels
        labels = ["2765HKR", "7784JGC", "7549LMH", "1296KSV", "2765HKR", "4135FNX", "1074KSN", "8622LW", "2765HKR",
                  "5479HHR", "3572CLX", "1842JJN", "6729HH", "7105GV", "9848HSZ", "3235MKB", "7718GZC", "1074KSN",
                  "2907HTR", "8622L", "4287KGJ", "1074KSN", "2976LY", "9101FLC", "5865LZT", "7362LWV", "5895KKT"]

        # Obtenim la predicció per cada model
