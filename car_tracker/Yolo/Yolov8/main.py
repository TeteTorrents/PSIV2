import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*

model=YOLO('yolov8s.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture(r'C:\Users\marti\Downloads\projecta\short.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

cotxe_down={}
contador_b=[]
cotxe_up={}
contador_p={}
count=0

tracker=Tracker()

cy1=150
cy2=350
offset=6

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = frame[frame.shape[0] // 2:, :]
    #frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
            list.append([x1,y1,x2,y2])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2

#Baixant Cotxes
        if cy1 < (cy+offset) and cy1 > (cy-offset):
            cotxe_down[id] = cy
        if id in cotxe_down:
            if cy2 < (cy+offset) and cy2 > (cy-offset):
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                if contador_b.count(id)==0:
                    contador_b.append(id)
#Pujant cotxes
        if cy2 < (cy+offset) and cy2 > (cy-offset):
            cotxe_up[id] = cy
        if id in cotxe_up:
            if cy1 < (cy+offset) and cy1 > (cy-offset):
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                if contador_p.count(id)==0:
                    contador_p.append(id)       


    cv2.line(frame,(135,cy1),(530,cy1),(255,255,0),1)
    cv2.line(frame,(70,cy2),(530,cy2),(255,0,255),1)
    c = (len(contador_b))
    p = (len(contador_p))
    cv2.putText(frame,('baixant: -')+str(c),(60,90),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    cv2.putText(frame,('pujant: -')+str(p),(60,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(2)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
