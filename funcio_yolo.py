"""modul que carrega el model preentrenat de yolo"""
from ultralytics import YOLO

def model(model):

    if model == 'bo':
        return YOLO(r'/workspaces/PSIV2/Yolov8/train4/weights/best.pt')
    
    if model == 'plate_detector':
        return YOLO(r'/workspaces/PSIV2/Yolov8/models/license_plate_detector.pt')
    
    else:
        return YOLO(r'/workspaces/PSIV2/Yolov8/yolov8n.pt')
    

def matricula (model, foto):

    results = model(foto)
    boxes = results.boxes.cpu().numy()
    x,y,z,t = boxes

    return x,y,z,t, boxes


