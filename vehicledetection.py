import cv2
import numpy as np
from time import sleep

largura_min = 80  # Minimum rectangle width
altura_min = 80   # Minimum rectangle height
offset = 6        # Allowed error in pixels
pos_linha = 550   # Counting line position
delay = 60        # Video FPS (frames per second)

# Real-world distance per pixel (calibration factor in meters per pixel)
meters_per_pixel = 0.05  # Adjust this based on the video
fps = delay  # Frames per second

# Variables
detec = []
carros = 0
vehicle_data = {}  # Store vehicle positions and speeds

def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv2.VideoCapture('video1.mp4')
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame1 = cap.read()
    if not ret:
        break

    tempo = float(1 / delay)
    sleep(tempo)

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = subtracao.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    contorno, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (255, 127, 0), 3)

    for (i, c) in enumerate(contorno):
        (x, y, w, h) = cv2.boundingRect(c)
        validar_contorno = (w >= largura_min) and (h >= altura_min)
        if not validar_contorno:
            continue

        centro = pega_centro(x, y, w, h)
        detec.append(centro)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame1, centro, 4, (0, 0, 255), -1)

        for (cx, cy) in detec:
            if cy < (pos_linha + offset) and cy > (pos_linha - offset):
                carros += 1
                cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (0, 127, 255), 3)

                # Calculate speed
                if cx not in vehicle_data:
                    vehicle_data[cx] = {'last_pos': cy, 'frame': carros}
                else:
                    last_pos = vehicle_data[cx]['last_pos']
                    last_frame = vehicle_data[cx]['frame']

                    # Calculate distance in pixels and convert to meters
                    distance_pixels = abs(cy - last_pos)
                    distance_meters = distance_pixels * meters_per_pixel

                    # Calculate time in seconds
                    time_seconds = (carros - last_frame) / fps

                    # Calculate speed (meters/second)
                    speed_mps = distance_meters / time_seconds if time_seconds > 0 else 0
                    speed_kmph = speed_mps * 3.6  # Convert to km/h

                    # Update vehicle data
                    vehicle_data[cx] = {'last_pos': cy, 'frame': carros}

                    # Display speed on the video near the vehicle
                    cv2.putText(frame1, f"{speed_kmph:.2f} km/h", (x + w + 10, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                detec.remove((cx, cy))

    # Display vehicle count
    cv2.putText(frame1, f"VEHICLE COUNT: {carros}", (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow("Video Original", frame1)
    cv2.imshow("Detectar", dilatada)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
