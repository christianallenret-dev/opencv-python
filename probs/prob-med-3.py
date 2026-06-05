import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image using the Haar cascade classifier
    # The detectMultiScale function detects objects (in this case, faces) in the input image and returns a list of rectangles where the detected objects are located.
    # The parameters (gray, 1.3, 5) specify the input image (grayscale), the scale factor (1.3) which determines how much the image size is reduced at each image scale, and the minimum number of neighbors (5) which specifies how many neighbors each candidate rectangle should have to retain it.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each detected face and apply a Gaussian blur 
    # to the region of interest (the detected face area) to anonymize it.
    for (x, y, w, h) in faces:
        # Extract the region of interest (the detected face area) from the original frame
        # This line takes the portion of the original frame that corresponds to the detected face (defined by the rectangle with top-left corner at (x, y) and width w and height h) and stores it in the variable 'roi' (region of interest).
        roi = frame[y:y+h, x:x+w]

        # Apply a Gaussian blur to the region of interest (the detected face area) to anonymize it
        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)

        # Replace the original face area in the frame with the blurred version to anonymize the detected face
        frame[y:y+h, x:x+w] = blurred_roi

        # Draw a white rectangle around the detected face area (the region of interest) to visually indicate where the face was detected and blurred
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Print the position of the detected face to the console (the x and y coordinates of the top-left corner of the detected face rectangle)
        print(f"Position: x={x}, y={y}")
    
    # Count the number of faces detected and print the count and position of each detected face to the console
    face_count = len(faces)
    cv2.putText(frame, f'Faces: {face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Face Blur", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
