import cv2
import os
import time

# Define the labels for each gesture
Labels = ["family", "hello", "help", "house", "i_love_you", "no", "please", "sorry", "thankyou","yes"]

# Set the number of images per label to 20
num_images_per_label = 50

# Create directories for each label if they don't exist
for label in Labels:
    os.makedirs(f'Tensorflow/workspace/images/collectedimages/{label}', exist_ok=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Start capturing images for each label
for label in Labels:
    print(f"Collecting images for '{label}' gesture.")
    img_count = 0

    while img_count < num_images_per_label:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame with instructions
        cv2.putText(frame, f"Label: {label} - Image {img_count}/{num_images_per_label}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to capture, 'q' to quit", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Image Capture", frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Press SPACE to capture the image
            img_path = f'Tensorflow/workspace/images/collectedimages/{label}/{img_count}.jpg'
            cv2.imwrite(img_path, frame)
            print(f"Captured image {img_count} for '{label}'")
            img_count += 1

        elif key == ord('q'):  # Press 'q' to exit early
            print("Exiting capture.")
            break

    # Add a 5-second delay before moving to the next label
    print("Waiting 5 seconds before moving to the next label...")
    time.sleep(5)

cap.release()
cv2.destroyAllWindows()
