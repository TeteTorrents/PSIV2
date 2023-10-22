import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import torch

model=YOLO('car_tracker/tracking_methods/optimized_yolo/yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('car_tracker/videos/short.mp4')

my_file = open("car_tracker/tracking_methods/optimized_yolo/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

count=0
tracker=Tracker()
offset=6
lower_region_x1 = 211
lower_region_y1 = 438
upper_region_x1 = 156
upper_region_y1 = 223

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 4 != 0:
        continue
    frame=cv2.resize(frame,(450,600))
    frame_lower = frame[438:539, 218:390]
    frame_upper = frame[223:293, 156:230]

    results_lower=model.predict(frame_lower)
    results_upper=model.predict(frame_upper)
    l=results_lower[0].boxes.data
    u=results_upper[0].boxes.data
    stacked_data = torch.cat((l,u), dim = 0)
    px=pd.DataFrame(stacked_data).astype("float")
    px['section'] = ['l' for i in range(len(l))] + ['u' for i in range(len(u))]
    list=[]
             
    for index,row in px.iterrows():
        
        if row['section'] == 'l':
            x1=int(row[0]) + lower_region_x1
            y1=int(row[1]) + lower_region_y1
            x2=int(row[2]) + lower_region_x1
            y2=int(row[3]) + lower_region_y1
        else:
            x1=int(row[0]) + upper_region_x1
            y1=int(row[1]) + upper_region_y1
            x2=int(row[2]) + upper_region_x1
            y2=int(row[3]) + upper_region_y1
        d=int(row[5])
        c=class_list[d]
        if 'car' in c or 'truck' in c or 'bus' in c:
            list.append([x1,y1,x2,y2])
    tracker.update(list)
    for id,bbox in tracker.center_points.items():
        cx=int(bbox[-1][0])
        cy=int(bbox[-1][1])
        cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
        cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

    print(tracker.center_points)
    print("---")     
    # Lower
    cv2.line(frame,(211,539),(390,539),(255,255,255),2)
    cv2.line(frame,(218,438),(377,438),(255,255,255),2)
    # Upperq
    cv2.line(frame,(156,293),(250,293),(0,255,255),2)
    cv2.line(frame,(163,223),(250,223),(0,255,255),2)
    cv2.putText(frame, f"PUJA: {tracker.up}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"BAIXA: {tracker.down}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()