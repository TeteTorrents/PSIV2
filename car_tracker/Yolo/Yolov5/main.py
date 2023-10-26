import cv2
import torch
from tracker import *
import numpy as np
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap=cv2.VideoCapture(r'car_tracker\videos\shadow.mp4')


def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

# To write result
size = (450, 600) 
result = cv2.VideoWriter('car_tracker/tracking_methods/sols/yolov5modshadow.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size) 


tracker = Tracker()
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(450,600))
    cv2.circle(frame, (140, 315), 4, (255, 255, 0), -1)
    results = model(frame)
    #frame = np.squeeze(results.render())
    list_res = []
    for idx,s in results.pandas().xyxy[0].iterrows():
        if (s['name'] == 'car' or s['name'] == 'truck' or s['name'] == 'bus') and s['confidence'] > 0.55:
            x1 = int(s['xmin'])
            y1 = int(s['ymin'])
            x2 = int(s['xmax'])
            y2 = int(s['ymax'])
            nom = str(s['name'])
            list_res.append([x1,y1,x2,y2])
            #cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 2)
            #cv2.putText(frame,nom,(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,255),1)
    boxes_ids = tracker.update(list_res)
    for box_id in boxes_ids:
        x,y,w,h,id = box_id
        cv2.rectangle(frame, (x,y), (w,h), (255,0,255), 2)
        cv2.putText(frame, str(id), (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(frame, f"PUJA: {tracker.up}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"BAIXA: {tracker.down}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('FRAME',frame)
    result.write(frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

result.release()
cap.release()
cv2.destroyAllWindows()