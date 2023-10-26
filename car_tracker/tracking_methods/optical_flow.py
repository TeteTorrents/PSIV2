import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

# Obrir video
cap = cv2.VideoCapture(r'car_tracker\videos\short.mp4')

# Parametres pel Lucas-Kanade Optical Flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners = 20, qualityLevel = 0.1, minDistance = 20, blockSize = 5)

# Altres parametres
trajectory_len = 20
detect_interval = 1
trajectories = []
vect_dir = {}
frame_idx = 0
cars_up = 0
cars_down = 0
can_execute = True
aux_frame = 0

# To write result
size = (450, 600) 
#result = cv2.VideoWriter('car_tracker/tracking_methods/sols/opf.avi',  
#                         cv2.VideoWriter_fourcc(*'MJPG'), 
#                         10, size) 


while True:

    suc, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()

    if frame_idx == aux_frame:
        can_execute = True

    # Calculate optical flow using Lucas-Kanade
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1

        new_trajectories = []
        new_vect_dir = {}

        # Get all the trajectories
        for idx, (trajectory, (x, y), good_flag) in enumerate(zip(trajectories, p1.reshape(-1, 2), good)):
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            try:
                new_vect_dir[idx].append(1 if (trajectory[-1][-1] - trajectory[-2][-1]) < 0 else -1)
            except:
                new_vect_dir[idx] = [1 if (trajectory[-1][-1] - trajectory[-2][-1]) < 0 else -1]
            # Newest detected point
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

            if int(y) == 480 and can_execute:
                latest_coords = np.array([trajectories[i][-1] for i in range(len(trajectories))])
                
                dbscan = DBSCAN(eps=50, min_samples=5)
                labels = dbscan.fit_predict(latest_coords)
                cluster_indices = defaultdict(list)
                
                for index, label in enumerate(labels):
                    if index != -1:
                        cluster_indices[label].append(index)
                cluster_indices = dict(cluster_indices)
                
                for c,v in cluster_indices.items():
                    c_center = np.mean(latest_coords[labels == c], axis = 0)
                    #print(sum([vect_dir[index][0] for index in cluster_indices[c] if index in vect_dir]))
                    #print([vect_dir[index][0] for index in cluster_indices[c] if index in vect_dir])
                    #print(c_center[1])
                    #print("--")
                    if c_center[1] > 480 + 50 or c_center[1] < 480 - 50 and c_center[0] > 180:
                        continue
                    if sum([vect_dir[index][0] for index in cluster_indices[c] if index in vect_dir]) > 0:
                        cars_up += 1
                    else:
                        cars_down += 1
                
                can_execute = False
                aux_frame = frame_idx + 100

        trajectories = new_trajectories
        vect_dir = new_vect_dir

        # Draw all the trajectories
        cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
    
    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        # Lastest point in latest trajectory
        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        # Detect the good features to track
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            # If good features can be tracked - add that to the trajectories
            aux_traj = []
            for x, y in np.float32(p).reshape(-1, 2):
                if y > 350 and x < 405 and x > 80:
                    #trajectories.append([(x, y)])
                    aux_traj.append((x,y))

    cluster_centers = []
    if aux_traj:
        dbscan = DBSCAN(eps=100, min_samples=3)
        cluster_labels = dbscan.fit_predict(np.array(aux_traj))
        for label in set(cluster_labels):
            if label != -1:  # Ignore noise points
                cluster_center = np.mean(np.array(aux_traj)[cluster_labels == label], axis=0)
                cluster_centers.append([tuple(cluster_center)])
    
    trajectories += cluster_centers

    frame_idx += 1
    prev_gray = frame_gray
    
    # Show Results
    cv2.putText(img, f"PUJA: {cars_up}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"BAIXA: {cars_down}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #cv2.imshow('OpF', img)
    #cv2.imshow('OpFMask', mask)
    frame_resized = cv2.resize(img, (450, 600))
    mask_resized = cv2.resize(mask, (450, 600))
    cv2.imshow('OpF', frame_resized)
    cv2.imshow('OpFMask', mask_resized)
    #result.write(frame_resized)
    print(frame_idx)
    #cv2.imshow('Mask', mask)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

#result.release()
cap.release()
cv2.destroyAllWindows()