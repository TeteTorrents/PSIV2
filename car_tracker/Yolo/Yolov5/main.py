import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from YOLOV5 import yolov5

# Inicialitzar el model YOLOv5
model = yolov5.Model('yolov5s')  

# Inicialitzar comptadors i diccionaris per al tracking
car_count = 0
tracked_cars = {}

# Funció per a calcular la distància euclidiana entre dos punts
def distancia_euclidiana(punt1, punt2):
    return np.sqrt((punt1[0] - punt2[0])**2 + (punt1[1] - punt2[1])**2)

# Funció per a fer el tracking dels cotxes
def trackejar_cotxes(frame):
    global car_count
    results = model(frame)  # Detectar objectes al frame

    # Iterar sobre els resultats de la detecció
    for det in results.xyxy[0].cpu().numpy():
        if det[4] > 0.3 and det[5] == 2:  # Filtrar deteccions de cotxes amb confiança > 0.3
            centre = ((det[0] + det[2]) / 2, (det[1] + det[3]) / 2)  # Calcular el centre del bounding box

            # Buscar cotxes propers i verificar el moviment a l'eix Y
            for car_id, tracked_car in tracked_cars.items():
                dist = distancia_euclidiana(centre, tracked_car['darrer_centre'])
                if dist < 50 and abs(centre[1] - tracked_car['darrer_centre'][1]) > 5:
                    if car_id not in tracked_car['ids_comptats']:
                        tracked_car['ids_comptats'].append(car_id)
                        if centre[1] > tracked_car['darrer_centre'][1]:
                            car_count += 1  # Cotxe pujant
                        else:
                            car_count -= 1  # Cotxe baixant

            # Afegir un nou cotxe al tracking
            tracked_cars[car_count] = {'darrer_centre': centre, 'ids_comptats': []}

            # Dibuixar el bounding box i l'ID del cotxe
            frame = cv2.rectangle(frame, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0, 255, 0), 2)
            frame = cv2.putText(frame, str(car_count), (int(centre[0]), int(centre[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame

# Processar un vídeo o frames individuals
cap = cv2.VideoCapture('/content/output7.mp4') 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Aplicar el tracking
    tracked_frame = trackejar_cotxes(frame)

    cv2_imshow(tracked_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Imprimir el total de cotxes comptats
print(f'Total de cotxes comptats: {car_count}')