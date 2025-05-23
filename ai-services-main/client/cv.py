import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read() 

    if not ret:
        print("Can't recieve frame")   

    frame = cv.flip(frame, 1)
    cv.imshow('frame', frame)
    
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()