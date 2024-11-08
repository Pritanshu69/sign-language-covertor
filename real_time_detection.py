# real_time_detection.py

import cv2
import numpy as np
import tensorflow as tf

Labels = ["hello", "thank_you", "yes", "no", "love_you"]
img_height, img_width = 64, 64

# Load the trained model
model = tf.keras.models.load_model('sign_language_model.h5')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (img_height, img_width))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    predictions = model.predict(img)
    predicted_label = Labels[np.argmax(predictions)]

    cv2.putText(frame, predicted_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
