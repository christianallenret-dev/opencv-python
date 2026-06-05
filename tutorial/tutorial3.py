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

    # Create a blank (black) image with the same size as the frame
    image = np.zeros(frame.shape, np.uint8)

    # Resize the frame to half its original size
    smaller_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Place rotated and original images into 4 quadrants:

    # Top-left: rotated 180 degrees (upside down)
    image[:height//2, :width//2] = cv2.rotate(smaller_frame, cv2.ROTATE_180)

    # Bottom-left: original smaller frame
    image[height//2:, :width//2] = smaller_frame

    # Top-right: rotated 180 degrees
    image[:height//2, width//2:] = cv2.rotate(smaller_frame, cv2.ROTATE_180)

    # Bottom-right: original smaller frame
    image[height//2:, width//2:] = smaller_frame

    # Show the final combined image in a window
    cv2.imshow('frame', image)

    # Wait 1 ms for a key press
    # If 'q' is pressed, exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()