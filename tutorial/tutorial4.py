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

    # Draw a blue diagonal line from top-left to bottom-right
    img = cv2.line(frame, (0, 0), (width, height), (255, 0, 0), 10)

    # Draw a green diagonal line from bottom-left to top-right
    img = cv2.line(img, (0, height), (width, 0), (0, 255, 0), 5)

    # Draw a gray rectangle from (100,100) to (200,200)
    img = cv2.rectangle(img, (100, 100), (200, 200), (128, 128, 128), 5)

    # Draw a filled red circle at position (300,300) with radius 60
    img = cv2.circle(img, (300, 300), 60, (0, 0, 255), -1)

    # Choose a font style for the text
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Put text on the image near the bottom-left corner
    img = cv2.putText(img, 'I am great!', (10, height - 10), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Show the final combined image in a window
    cv2.imshow('frame', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()