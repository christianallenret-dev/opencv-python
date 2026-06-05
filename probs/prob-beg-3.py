import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    width = int(cap.get(3))
    height = int(cap.get(4))

    img = cv2.rectangle(frame, (50, 50), (200, 200), (0, 0, 255), (3))

    img = cv2.circle(img, (width // 2, height // 2), 40, (0, 255, 0), -1)

    img = cv2.putText(img, 'OpenCV is fun!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Drawing', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()