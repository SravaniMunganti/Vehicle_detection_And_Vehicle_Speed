import cv2
import numpy as np
from math import sqrt
from time import sleep

# Parameters
largura_min = 80  # Minimum rectangle width
altura_min = 80   # Minimum rectangle height
offset = 6        # Allowed error in pixel count
pos_linha = 550   # Line position for counting
fps = 30          # Video frame rate (frames per second)
scaling_factor = 0.05  # Real-world distance per pixel (meters per pixel, change this as per your video)

# Initialize variables
detec = []
carros = 0
vehicle_positions = {}

# Helper function to calculate the center of a rectangle
def pega_centro(x, y, w, h):
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return cx, cy

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# Open the video file
cap = cv2.VideoCapture('video1.mp4')
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Background subtraction
subtracao = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame1 = cap.read()
    if not ret or frame1 is None:
        print("End of video or error reading frame.")
        break

    # Preprocess the frame
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = subtracao.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the counting line
    cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (255, 127, 0), 3)

    # Process contours
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        validar_contorno = (w >= largura_min) and (h >= altura_min)
        if not validar_contorno:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        centro = pega_centro(x, y, w, h)

        # Add or update vehicle position
        vehicle_detected = False
        for vehicle_id, prev_position in list(vehicle_positions.items()):
            distance = calculate_distance(prev_position, centro)
            if distance < 50:  # If close to a previously detected vehicle
                # Calculate speed (distance in meters / time in seconds)
                real_distance = distance * scaling_factor  # Convert to meters
                speed = real_distance * fps * 3.6  # Convert to km/h

                # Display speed on the video screen
                speed_text = f"Speed: {speed:.2f} km/h"
                cv2.putText(frame1, speed_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Update the vehicle's position
                vehicle_positions[vehicle_id] = centro
                vehicle_detected = True
                break

        if not vehicle_detected:
            # Assign a new ID for a new vehicle
            vehicle_id = len(vehicle_positions) + 1
            vehicle_positions[vehicle_id] = centro

        detec.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0, 255), -1)

        for (cx, cy) in detec:
            if (cy < (pos_linha + offset)) and (cy > (pos_linha - offset)):
                carros += 1
                cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (0, 127, 255), 3)
                detec.remove((cx, cy))
                print(f"Car detected: {carros}")

    # Display the frame and vehicle count
    cv2.putText(frame1, f"VEHICLE COUNT: {carros}", (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame1)
    cv2.imshow("Detection", dilatada)

    # Exit on 'ESC' key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
