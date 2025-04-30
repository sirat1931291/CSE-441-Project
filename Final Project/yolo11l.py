import cv2
from ultralytics import YOLO

model = YOLO("yolo11l.pt")

video_path = "cctv1.mp4" 
cap = cv2.VideoCapture(video_path)

roi_top_left = (0, 0)
roi_bottom_right = (800, 500)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    person_count = 0

    cv2.rectangle(frame, roi_top_left, roi_bottom_right, (255, 0, 0), 2)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if (roi_top_left[0] <= center_x <= roi_bottom_right[0] and
                    roi_top_left[1] <= center_y <= roi_bottom_right[1]):
                    person_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 3, (0, 255, 255), -1)
                    cv2.putText(frame, "Person", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    text = f"People in ROI: {person_count}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = frame.shape[1] - text_size[0] - 10
    text_y = frame.shape[0] - 20
    cv2.putText(frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("YOLOv11 - People in ROI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
