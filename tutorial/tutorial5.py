import cv2
import numpy as np

# Open the default camera (0 = built-in webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() # Capture a frame from the camera
    # ret = True if frame was captured successfully
    # frame = the actual image

    width = int(cap.get(3))  # Get frame width
    height = int(cap.get(4)) # Get frame height

    # Convert the image from BGR (default color) to HSV (better for color detection)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of blue color in HSV
    lower_blue = np.array([90, 50, 50])     # lower bound of blue
    upper_blue = np.array([130, 255, 255])  # upper bound of blue

    # Create a mask that keeps only blue colors and removes others
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply the mask to the original image
    # This keeps only the blue parts and makes everything else black
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the result (only blue parts visible)
    cv2.imshow('frame', result)

    # Display the mask (white = detected color, black = ignored)
    cv2.imshow('mask', mask)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()