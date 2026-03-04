import cv2
from ultralytics import YOLO
 
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
 
cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Detection", 800, 600)    

while True:
    ret, frame = cap.read()
    if not ret:
        break

     
    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Detection", annotated_frame)
 
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()