import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('webcame', frame)

    key = cv2.waitKey(1) & 0xFF

    # Press s to get snapshot
    if key == ord('s'):
        filename = 'snapshot.jpg'
        cv2.imwrite(filename, frame)
        print(f'Saved {filename}')

    # Press q to exit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()