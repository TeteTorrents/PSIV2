import cv2
import numpy as np

# Obrir video
cap = cv2.VideoCapture(r'car_tracker\videos\shadow.mp4')

# Inicialitzem el substractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Definim parametres i variables
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) 
min_contour_area = 2500
cars = {}
cars_timeouts = {}
vect_dir = {}
car_directions = {}
car_id_counter = 0
delay_frame = 75
frame_count = {}
frame_idx = 0
car_timeout_threshold = 100
cars_up, cars_down = 0, 0

# To write result
#size = (450, 600) 
#result = cv2.VideoWriter('car_tracker/tracking_methods/sols/gmm.avi',  
#                         cv2.VideoWriter_fourcc(*'MJPG'), 
#                         10, size) 

# Iterem pels diferents frames
while True:

    # Llegim el frame
    suc, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apliquem GMM background substraction al frame + netegem
    fgmask = fgbg.apply(value_channel)
    mask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Trobem els contorns + Ignorem la part superior del video
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area and cv2.boundingRect(cnt)[1] > 350 and cv2.boundingRect(cnt)[0] > 80 and cv2.boundingRect(cnt)[0] < 405]

    # Diccionari pels cotxes nous
    new_cars = cars

    # Actualitzem temps de desapareinxa dels cotxes
    for car_id in cars.keys():

        if car_id not in car_directions:
            car_directions[car_id] = "undetermined"
            frame_count[car_id] = 0

        if car_id in cars:
            cars_timeouts[car_id] += 1
        else:
            cars_timeouts[car_id] = 1

    for contour in valid_contours:
        # Calculem el centroide
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Mirem si hi ha algun cotxe prop del centroide
            closest_car_id = None
            min_distance = 100 # Això s'ha de tunejar

            for car_id, car_info in cars.items():
                last_centroid = car_info['centroid']
                distance = np.linalg.norm(np.array(last_centroid) - np.array((cx, cy)))
                
                if distance < min_distance:
                    min_distance = distance
                    closest_car_id = car_id
            
            # Si hi ha cotxe prop dels identificadors, actualitzem el seu valor
            if closest_car_id is not None:
                aux = cars[closest_car_id]['centroid'][1]
                new_cars[closest_car_id] = {'centroid': (cx, cy)}
                cars_timeouts[closest_car_id] = 0 # Fem reset si encara es detecta el cotxe
                vect_dir[closest_car_id].append(1 if (new_cars[closest_car_id]['centroid'][1] - aux) < 0 else -1)
            else: # en cas contrari creem nou cotxe
                if cx > 200 and (cy > 550 or cy < 430):
                    new_car_id = car_id_counter + 1
                    car_id_counter += 1
                    new_cars[new_car_id] = {'centroid': (cx, cy)}
                    cars_timeouts[new_car_id] = 0
                    vect_dir[new_car_id] = []
            
            #print(int(cy))
            if int(cy) in [i for i in range(550, 565)]:
                if closest_car_id is not None:
                    if sum(vect_dir[closest_car_id][int(len(vect_dir[closest_car_id])/1.3):]) > 0:
                        if car_directions[closest_car_id] == "down":
                            frame_count[closest_car_id] = 1
                        else:
                            frame_count[closest_car_id] += 1
                        
                        if frame_count[closest_car_id] >= delay_frame or frame_count[closest_car_id] == 1:
                            if car_directions[car_id] == "down":
                                cars_down -= 1
                            car_directions[car_id] = "up"
                            cars_up += 1
                    else:
                        if car_directions[closest_car_id] == "up":
                            frame_count[closest_car_id] = 1
                        else:
                            frame_count[closest_car_id] += 1
                        
                        if frame_count[closest_car_id] >= delay_frame or frame_count[closest_car_id] == 1:
                            if car_directions[car_id] == "up":
                                cars_up -= 1
                            car_directions[car_id] = "down"
                            cars_down += 1
    
    # Eliminem els cotxes que no es veuen/detecten desde fa molt
    cars_to_remove = [car_id for car_id, timeout in cars_timeouts.items() if timeout >= car_timeout_threshold]
    for car_id in cars_to_remove:
        cars.pop(car_id)
        cars_timeouts.pop(car_id)

    cars = new_cars.copy()

    cv2.circle(frame, (150, 550), 8, (255, 255, 0), -1)
    for car_id, car_info in cars.items():
        cx, cy = car_info['centroid']
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"id:{car_id}", (cx+5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"PUJA: {cars_up}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"BAIXA: {cars_down}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    frame_resized = cv2.resize(frame, (450, 600))
    vc_resized = cv2.resize(value_channel, (450, 600))
    mask_resized = cv2.resize(mask, (450, 600))
    print(frame_idx)
    frame_idx += 1
    cv2.imshow('frame', frame_resized)
    #result.write(frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#result.release()
cap.release()
cv2.destroyAllWindows()