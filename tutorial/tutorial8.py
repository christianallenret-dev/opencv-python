import numpy as np
import cv2

# Open the default camera (0 = built-in webcam)
cap = cv2.VideoCapture(0)

# Load pre-trained classifiers for detecting faces and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    # Capture a frame (image) from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale (needed for detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Draw a blue rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

        # Get the region of interest (face area) in grayscale and color
        roi_gray = gray[y:y+h, x:x+w]   # NOTE: fixed slicing bug here
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        # Loop through each detected eye
        for (ex, ey, ew, eh) in eyes:
            # Draw a green rectangle around each eye
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

    # Show the video frame with detections
    cv2.imshow('Frame', frame)

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()