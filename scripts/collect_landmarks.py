# scripts/collect_landmarks.py
import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils


DATASET_PATH = r"C:\convoease\dataset\train"  


X, y = [], []


labels = sorted(os.listdir(DATASET_PATH))
print("Labels found:", labels)


for idx, label in enumerate(labels):
    folder_path = os.path.join(DATASET_PATH, label)
    images = os.listdir(folder_path)
    print(f"\nProcessing folder: {folder_path}, Found {len(images)} image files")

    for i, img_name in enumerate(images):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Cannot read {img_path}")
            continue

        
        img = cv2.resize(img, (256, 256))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            landmarks = []
            for lm in result.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            X.append(landmarks)
            y.append(idx)
        else:

            if i % 100 == 0:
                print(f"No hand detected in {img_name}")

        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} / {len(images)} images in label '{label}'")


X = np.array(X)
y = np.array(y)


np.save("X_landmarks.npy", X)
np.save("y_labels.npy", y)

print(f"\nSaved landmarks. X shape: {X.shape}, y shape: {y.shape}")
