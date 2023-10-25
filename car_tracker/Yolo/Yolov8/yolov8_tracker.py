import cv2
from ultralytics import YOLO
import pandas as pd 
import torch   

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

my_file = open(r"C:\Users\marti\Downloads\soc tonto\.venv\PSIV2\car_tracker\Yolo\Yolov8\coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 


cotxe_down={}
contador_b=[]
cotxe_up={}
contador_p=[]
count=0


cy1=150
cy2=350
offset=6

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')


# Open the video file
video_path = r"C:\Users\marti\Downloads\soc tonto\.venv\PSIV2\car_tracker\videos\short.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame = frame[frame.shape[0] // 2:, :]

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes = 2 )

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.line(annotated_frame,(135,cy1),(530,cy1),(255,255,0),1)
        cv2.line(annotated_frame,(70,cy2),(530,cy2),(255,0,255),1)

        for track in results[0].boxes.xyxy:
            # Obtiene el centro del coche
            cy = track[1].numpy()
            print(cy)

            while True:
                if cy1 - offset < cy < cy1 + offset:
                    contador_p.append(1)
                    break

                if cy2 - offset < cy < cy2 + offset:
                    contador_b.append(1)
                    print(contador_p)
                    break
                
                else:
                    break

        c = (len(contador_b))
        p = (len(contador_p))
        cv2.putText(annotated_frame,('baixant: -')+str(c),(60,90),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
        cv2.putText(annotated_frame,('pujant: -')+str(p),(60,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
