import cv2
import numpy as np

# Load the image from file
img = cv2.imread('assets/chessboard.png')

# Resize the image to 75% of its original size
img = cv2.resize(img, (0, 0), fx=0.75, fy=0.75)

# Convert the image to grayscale (needed for corner detection)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect up to 100 strong corners in the image
# 0.01 = quality level (lower = more corners)
# 10 = minimum distance between detected corners
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)

# Convert corner coordinates to integers
corners = np.intp(corners)

# Draw a blue circle on each detected corner
for corner in corners:
    x, y = corner.ravel()  # Get x and y coordinates
    cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

# Draw lines between every pair of corners
for i in range(len(corners)):
    for j in range(i + 1, len(corners)):
        corner1 = tuple(corners[i][0])  # First corner
        corner2 = tuple(corners[j][0])  # Second corner
        
        # Generate a random color for each line
        color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3)))
        
        # Draw the line between the two corners
        cv2.line(img, corner1, corner2, color, 1)

# Display the final image in a window
cv2.imshow("Frame", img)

# Wait until a key is pressed
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()