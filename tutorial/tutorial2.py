import cv2
import random

img = cv2.imread('assets/meme.jpg', 1)

# This block (currently commented out) would add random colors
# to the top 100 rows of the image (creates a noise effect)
"""
for i in range(100):
    for j in range(img.shape[1]):
        img[i][j] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
"""

# Extract (crop) a small rectangular region from the image
# Rows 100–199 and columns 50–99
tag = img[100:200, 50:100]

# Paste that cropped region into a new location
# Rows 100–199 and columns 175–224
# (Same size as 'tag', so assignment works)
img[100:200, 175:225] = tag

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()