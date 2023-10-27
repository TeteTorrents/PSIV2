from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = 'short.mp4'
cap = cv2.VideoCapture(video_path)

# Write video
size = (450, 600) 
result = cv2.VideoWriter('short.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size) 

# Store the track history + counting
track_history = defaultdict(lambda: [])
car_counted = set()

# Counting up or down
cars_up = 0
cars_down = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        frame = frame[frame.shape[0] // 2:, :]
        results = model.track(frame, persist=True, classes = 2)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.int().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        #cv2.line(annotated_frame, (135,150), (530,150), (0,255,0), 2)

        # Plot the tracks
        for box, track_id, cls in zip(boxes, track_ids, classes):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Count
            if int(y) in [i for i in range(460,480)] and (cls == 2 or cls == 7):
                prev_y = track_history[track_id][-2][1]
                direction = "up" if y < prev_y else "down"

                if direction == "up" and track_id not in car_counted:
                    car_counted.add(track_id)
                    cars_up += 1
                elif direction == "down" and track_id not in car_counted:
                    car_counted.add(track_id)
                    cars_down += 1

            # Draw the tracking lines
            #points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            #cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Display the annotated frame
        cv2.putText(annotated_frame, f"PUJA: {cars_up}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"BAIXA: {cars_down}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_resized = cv2.resize(annotated_frame, (450, 600))
        result.write(frame_resized)
        cv2.imshow("YOLOv8 Tracking", frame_resized)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
result.release()
cap.release()
cv2.destroyAllWindows()
