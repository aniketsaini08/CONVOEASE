# scripts/predict_camera.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp


model = load_model("hand_gesture_model.h5")


labels = sorted(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", 
                 "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", 
                 "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"])


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1) 
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            X_input = np.array([landmarks])

            
            pred = model.predict(X_input, verbose=0)
            gesture = labels[np.argmax(pred)]

            
            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 3)

   
    cv2.imshow("Hand Gesture Recognition", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
