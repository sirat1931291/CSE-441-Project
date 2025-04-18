import torch
import cv2

# Load YOLOv5s model (only detect people)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0]  # Only class 0 = 'person'

# Load your video
# cap = cv2.VideoCapture("cctv1.mp4")
cap = cv2.VideoCapture(0)

# Define region of interest (ROI) – (x1, y1), (x2, y2)
roi_top_left = (0, 0)
roi_bottom_right = (800, 500)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = frame.copy()

    person_count = 0

    # Draw the ROI rectangle
    cv2.rectangle(annotated_frame, roi_top_left, roi_bottom_right, (255, 0, 0), 2)

    for det in results.pred[0]:
        if int(det[5]) == 0:  # Person class
            x1, y1, x2, y2 = map(int, det[:4])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Check if person is inside ROI
            if (roi_top_left[0] <= center_x <= roi_bottom_right[0] and
                roi_top_left[1] <= center_y <= roi_bottom_right[1]):
                
                person_count += 1
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(annotated_frame, (center_x, center_y), 3, (0, 255, 255), -1)
                cv2.putText(annotated_frame, 'Person', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show count
    # cv2.putText(annotated_frame, f"People in ROI: {person_count}", (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(annotated_frame, f"People in ROI: {person_count}", 	(10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("YOLOv5 - People in ROI", annotated_frame)

    # Keep original speed
    if cv2.waitKey(int(1000 / cap.get(cv2.CAP_PROP_FPS))) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
