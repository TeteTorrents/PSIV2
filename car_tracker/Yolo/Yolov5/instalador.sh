#!/bin/bash

# Instala el gestor de paquets Pip
pip install --upgrade pip

# Clona el repositori YOLOv5
git clone https://github.com/ultralytics/yolov5

# Entra al directori del repositori
cd yolov5

# Instala la llibreria YOLOv5
pip install -e .

# Confirma que la llibreria s'ha instal·lat correctament
import yolov5
