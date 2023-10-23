import cv2
import pandas as pd
from yolov5 import models, tracking
import torch

model = models.yolov5s()
model.load("yolov5s.pt")

cap = cv2.VideoCapture("car_tracker/videos/short.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (450, 600))
    frame_lower = frame[438:539, 218:390]
    frame_upper = frame[223:293, 156:230]

    results_lower = model(frame_lower)
    results_upper = model(frame_upper)

    l = results_lower[0].boxes.data
    u = results_upper[0].boxes.data
    stacked_data = torch.cat((l, u), dim=0)
    px = pd.DataFrame(stacked_data).astype("float")
    px["section"] = ["l" for i in range(len(l))] + ["u" for i in range(len(u))]

    list = []
    for index, row in px.iterrows():
        if row["section"] == "l":
            x1 = int(row[0]) + lower_region_x1
            y1 = int(row[1]) + lower_region_y1
            x2 = int(row[2]) + lower_region_x1
            y2 = int(row[3]) + lower_region_y1
        else:
            x1 = int(row[0]) + upper_region_x1
            y1 = int(row[1]) + upper_region_y1
            x2 = int(row[2]) + upper_region_x1
            y2 = int(row[3]) + upper_region_y1
        list.append([x1, y1, x2, y2])

    tracker = tracking.Tracker()
    tracker.update(list)

    for id, bbox in tracker.center_points.items():
        cx = int(bbox[-1][0])
        cy = int(bbox[-1][1])
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.line(frame, (211, 539), (390, 539), (255, 255, 255), 2)
    cv2.line(frame, (218, 438), (377, 438), (255, 255, 255), 2)
    cv2.line(frame, (156, 293), (250, 293), (0, 255, 255), 2)
    cv2.line(frame, (163, 223), (250, 223), (0, 255, 255), 2)
    cv2.putText(frame, f"PUJA: {tracker.up}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"BAIXA: {tracker.down}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
