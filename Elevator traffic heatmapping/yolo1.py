import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load downloaded video
cap = cv2.VideoCapture("cctv1.mp4")  # Change to your video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    # annotated_frame = results.render()[0]
    annotated_frame = results.render()[0].copy()  # Now it's writable


    # Count people (class 0 = person)
    person_count = sum([1 for x in results.pred[0] if int(x[5]) == 0])
    cv2.putText(annotated_frame, f"People: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("YOLOv5 People Detection", annotated_frame)

    # Delay based on FPS
    if cv2.waitKey(int(1000 / cap.get(cv2.CAP_PROP_FPS))) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
