import cv2

# Reads/grabs the image from the 'assets' folder
img = cv2.imread('assets/meme.jpg', 0) # '0' is the default mode for loading image

# Resize the img
img = cv2.resize(img, (0, 0), fx=2, fy=2)

# Rotates the img
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Creates a new img file
cv2.imwrite('new_img.jpg', img)

# Shows the img
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()