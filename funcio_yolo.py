import torch
from torchvision.transforms import functional as F
from torchvision import models
from PIL import Image
import cv2
import numpy as np

def carregar_model():
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def carregar_yolo():
    model_yolo = YOLO('best.pt')
    # Aquí has de carregar el teu model YOLO prèviament entrenat
    # Necessitaràs tenir les funcions necessàries per carregar i utilitzar el model YOLO.
    # Per exemple, pots utilitzar les biblioteques 'darknet' o 'pytorch-YOLOv3'.
    # El codi per carregar el model YOLO pot variar segons la biblioteca que estiguis utilitzant.

    # Retorna el model YOLO carregat
    return la_teu_funcio_per_carregar_yolo()

def detectar_matricula(imatge, model_yolo):
    # Funció per detectar la matrícula en una imatge
    # imatge: ruta de la imatge o una matriu d'imatge (numpy array)
    # model_yolo: el model YOLO carregat

    # Si l'entrada és una ruta d'imatge, la carreguem
    if isinstance(imatge, str):
        imatge = cv2.imread(imatge)

    # Preprocessament de la imatge
    # Això pot variar depenent de com hagi estat entrenat el teu model YOLO
    # Assegura't que la imatge estigui en el format correcte (per exemple, BGR o RGB)
    # i que les dimensions coincideixin amb les expectatives del model

    # Realitza la detecció amb YOLO
    resultats_yolo = model_yolo.detect(imatge)

    # Filtra els resultats per obtenir les deteccions de matrícules
    matricules = filtrar_resultats(resultats_yolo)

    return matricules

def filtrar_resultats(resultats_yolo):
    # Aquesta funció filtra els resultats de YOLO per obtenir les deteccions de matrícules
    # Pots necessitar adaptar aquesta funció depenent de com sigui el format dels resultats de YOLO

    # Per exemple, si els resultats de YOLO tenen una llista de deteccions en la forma:
    # [(classe, probabilitat, bounding_box), ...]
    # llavors podríes filtrar per classe == "matricula" o per alguna altra etiqueta que hagis utilitzat per entrenar.

    # Retorna una llista de bounding boxes de matrícules
    return [(classe, probabilitat, bounding_box) for classe, probabilitat, bounding_box in resultats_yolo if classe == "matricula"]

def visualitzar_resultats(imatge, matricules):
    # Funció per visualitzar els resultats
    # imatge: ruta de la imatge o una matriu d'imatge (numpy array)
    # matricules: llista de bounding boxes de matrícules

    if isinstance(imatge, str):
        imatge = cv2.imread(imatge)

    for _, _, bounding_box in matricules:
        x, y, w, h = bounding_box
        cv2.rectangle(imatge, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Resultat', imatge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ús de les funcions

model_yolo = carregar_yolo()
imatge = "ruta_de_la_teva_imatge.jpg"
matricules = detectar_matricula(imatge, model_yolo)
visualitzar_resultats(imatge, matricules)
