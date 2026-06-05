import cv2

cap = cv2.VideoCapture(0)

# Read a reference frame from the video feed to use as a baseline for motion detection
# This reference frame will be used to compare against subsequent frames to detect any changes (motion) in the scene.
ret, ref_frame = cap.read()

# Convert the reference frame to grayscale, which simplifies the image and makes it easier to detect changes in intensity (motion) between frames.
ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

while True:
    # Read a new frame from the video feed to compare against the reference frame for motion detection
    ret, frame = cap.read()

    if not ret:
        break
    
    # Convert the current frame to grayscale to prepare it for motion detection by comparing it with the reference grayscale frame.
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the reference grayscale frame and the current grayscale frame to 
    # identify areas of change (motion) in the scene.
    diff = cv2.absdiff(ref_gray, grey)

    # Apply a binary threshold to the difference image to create a binary mask 
    # that highlights areas of significant change (motion) in the scene.
    # The threshold value of 25 means that any pixel with a difference greater than 25 
    # will be set to 255 (white) in the binary mask, while others will be set to 0 (black).
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded image to fill in gaps and make the contours of the detected motion more pronounced,
    # which helps in accurately identifying and drawing bounding boxes around the areas of motion.
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours in the dilated image, which represent the areas of motion detected in the scene.
    countours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a flag to indicate whether motion has been detected in the current frame. 
    # This will be set to True if any contours (areas of motion) are found that exceed a certain size threshold.
    motion_detected = False

    # Loop through the detected contours to check if any of them represent significant motion 
    # (i.e., if their area exceeds a certain threshold).
    for c in countours:
        # Check if the area of the contour is greater than 800 pixels to filter out small movements or noise 
        # that may not be considered significant motion.
        if cv2.contourArea(c) > 800:
            # If a significant contour is found, draw a bounding rectangle around it on the original frame to visually indicate where motion has been detected.
            x_rect, y_rect, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x_rect, y_rect), (x_rect + w, y_rect + h), (0, 255, 0), 2)
            motion_detected = True
    
    # If motion is detected (i.e., if any contours were found that exceed the area threshold), print a message to the console indicating that motion has been detected.
    if motion_detected:
        print("Motion detected!")
    
    # Display the original frame with any detected motion highlighted by bounding rectangles, 
    # and also display the thresholded image that shows the areas of motion in white.
    cv2.imshow("Frame", frame)

    # Display the thresholded image (dilated) to visualize the areas of motion detected in the scene, 
    # which are highlighted in white.
    cv2.imshow("Threshold", dilated)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

    