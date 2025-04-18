import torch
import cv2

# Load YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0]  # 👈 ONLY detect class 0 = 'person'

# Load your video
cap = cv2.VideoCapture("cctv1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Make the frame writable
    annotated_frame = frame.copy()

    # Draw only person detections manually
    person_count = 0
    for det in results.pred[0]:
        if int(det[5]) == 0:  # class 0 = person
            person_count += 1
            x1, y1, x2, y2 = map(int, det[:4])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, 'Person', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show person count
    cv2.putText(annotated_frame, f"People: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("YOLOv5 - People Only", annotated_frame)

    # Maintain original video speed
    if cv2.waitKey(int(1000 / cap.get(cv2.CAP_PROP_FPS))) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
