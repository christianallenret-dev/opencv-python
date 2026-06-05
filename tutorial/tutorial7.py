import numpy as np
import cv2

# Load the main image (grayscale) and resize it to 80% of original size
img = cv2.resize(cv2.imread('assets/soccer_practice.jpg', 0), (0, 0), fx=0.8, fy=0.8)

# Load the template image (grayscale) and resize it as well
template = cv2.resize(cv2.imread('assets/shoe.png', 0), (0, 0), fx=0.8, fy=0.8)

# Get height and width of the template image
h, w = template.shape

# List of different template matching methods to test
methods = [
    cv2.TM_CCOEFF, 
    cv2.TM_CCOEFF_NORMED, 
    cv2.TM_CCORR, 
    cv2.TM_CCORR_NORMED, 
    cv2.TM_SQDIFF, 
    cv2.TM_SQDIFF_NORMED
]

# Loop through each matching method
for method in methods:
    # Make a copy of the original image (so we don’t overwrite it)
    img2 = img.copy()
    
    # Apply template matching using the selected method
    result = cv2.matchTemplate(img2, template, method)
    
    # Get the minimum and maximum matching values and their locations
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # For SQDIFF methods, the best match is the *minimum* value
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        # For other methods, the best match is the *maximum* value
        location = max_loc

    # Calculate bottom-right corner of the matched region
    bottom_right = (location[0] + w, location[1] + h)

    # Draw a rectangle around the detected match
    cv2.rectangle(img2, location, bottom_right, 255, 5)

    # Show the result in a window
    cv2.imshow('Match', img2)

    # Wait until a key is pressed
    cv2.waitKey(0)

    # Close the window before showing the next result
    cv2.destroyAllWindows()