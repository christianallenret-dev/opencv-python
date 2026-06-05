import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the image from BGR (default color) to HSV (better for color detection)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a mask that keeps only yellow colors and removes others
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply the mask to the original image
    # This keeps only the yellow parts and makes everything else black
    # 'frame' is the original image, 'mask' is the binary mask where yellow parts are white and others are black
    # 'mask=mask' means we want to apply the mask to the original image, keeping only the yellow parts
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Find contours in the mask (the white areas representing detected yellow objects)
    # Contours are the boundaries of the detected objects in the mask, and hierarchy provides information about the contour structure (e.g., nested contours)
    # In short, this line detects the contours of the detected yellow objects in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found (i.e., if any yellow objects were detected)
    if contours:
        # Find the largest contour among the detected contours, which is likely to be the most significant yellow object in the frame
        # key=cv2.contourArea tells the max function to compare contours based on their area, 
        # so it will return the contour with the largest area (the largest detected yellow object)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding rectangle of the largest contour and draw it on the original frame
        # cv2.boundingRect() returns the x and y coordinates of the top-left corner of the rectangle, 
        # and its width (w) and height (h)
        # We then use cv2.rectangle() to draw a rectangle on the original frame using these coordinates and dimensions
        x_rect, y_rect, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x_rect, y_rect), (x_rect + w, y_rect + h), (255, 0, 0), 2)

        # Calculate the moments of the largest contour to find its center
        M = cv2.moments(largest_contour)

        # Check if the area of the contour (M['m00']) is not zero to avoid division by zero when calculating the center
        if M['m00'] != 0:
            # Calculate the center of the detected object using the moments of the contour
            # The center (cX, cY) is calculated using the spatial moments of the contour.
            # M['m10'] and M['m01'] are the first-order moments, and M['m00'] is the zeroth-order moment (area of the contour).
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            # Draw a red circle at the center of the detected object (the center of the contour)
            # cv2.circle() is used to draw a circle on the original frame at the coordinates (cX, cY) with a radius of 8 pixels and a color of red (0, 0, 255 in BGR format)
            cv2.circle(frame, (cX, cY), 8, (0, 0, 255), -1)

            # Print the coordinates of the detected object (the center of the contour) to the console
            print(f"Object detected at: ({cX}, {cY})")

    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()