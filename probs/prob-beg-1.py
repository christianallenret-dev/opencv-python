import cv2

img = cv2.imread('assets/meme.jpg', 1) # '1' is color mode, while '0' is greyscale mode

# ORDER OF VARIABLES IS IMPORTANT
height, width, channels = img.shape # gets width, height, and color channel of image in exact order

print(f"Width: {width}")
print(f"Height: {height}")
print(f"Channels: {channels}")

flipped_h = cv2.flip(img, 1) # flips image vertically
flipped_v = cv2.flip(img, 0) # flips image horizontally
flipped_both = cv2.flip(img, -1) # flips on both sides

cv2.imshow('Vertical', flipped_v)
cv2.imshow('Horizontal', flipped_h)
cv2.imshow('Both Flipped', flipped_both)
cv2.waitKey(0)
cv2.destroyAllWindows()
