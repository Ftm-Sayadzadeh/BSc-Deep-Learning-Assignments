import cv2
from ultralytics import YOLO

model = YOLO('weights/best.pt')

cap = cv2.VideoCapture(0)

print("enter Q for exit!")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    # imgsz=640
    # conf=0.5  
    results = model(frame, conf=0.5, imgsz=640)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Real-time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()