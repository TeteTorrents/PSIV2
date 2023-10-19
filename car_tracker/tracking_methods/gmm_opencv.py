import cv2
import numpy as np

# Obrir video
cap = cv2.VideoCapture(r'car_tracker\videos\short.mp4')

# Inicialitzem el substractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Definim parametres
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) 
min_contour_area = 2000
cars = {}
cars_timeouts = {}
car_timeout_threshold = 50

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
    """
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    mask = cv2.erode(mask, kernel2, iterations = 1)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
    mask = cv2.erode(mask, kernel3, iterations = 1)
    kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.dilate(mask, kernel4, iterations = 3)
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
    mask = cv2.dilate(mask, kernel5, iterations = 3)
    """

    # Trobem els contorns + Ignorem la part superior del video
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area and cv2.boundingRect(cnt)[1] > 350 and cv2.boundingRect(cnt)[0] > 80 and cv2.boundingRect(cnt)[0] < 405]

    # Diccionari pels cotxes nous
    new_cars = cars

    # Actualitzem temps de desapareinxa dels cotxes
    for car_id in cars.keys():
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
                new_cars[closest_car_id] = {'centroid': (cx, cy)}
                cars_timeouts[closest_car_id] = 0 # Fem reset si encara es detecta el cotxe
            else: # en cas contrari creem nou cotxe
                if cx > 200 and (cy > 530 or cy < 430):
                    new_car_id = len(cars) + 1
                    new_cars[new_car_id] = {'centroid': (cx, cy)}
                    cars_timeouts[new_car_id] = 0
    
    # Eliminem els cotxes que no es veuen/detecten desde fa molt
    cars_to_remove = [car_id for car_id, timeout in cars_timeouts.items() if timeout >= car_timeout_threshold]
    for car_id in cars_to_remove:
        cars.pop(car_id)
        cars_timeouts.pop(car_id)

    cars = new_cars.copy()

    #cv2.circle(frame, (150, 350), 8, (255, 255, 0), -1)
    for car_id, car_info in cars.items():
        cx, cy = car_info['centroid']
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    print(cars)

    cv2.imshow('frame', frame)
    cv2.imshow('vc', value_channel)
    cv2.imshow('mask', mask)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()