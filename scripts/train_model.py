# scripts/train_model.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


X = np.load("X_landmarks.npy")
y = np.load("y_labels.npy")
y = to_categorical(y) 


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(y.shape[1], activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)


model.save("hand_gesture_model.h5")
print("Model trained and saved as hand_gesture_model.h5")
