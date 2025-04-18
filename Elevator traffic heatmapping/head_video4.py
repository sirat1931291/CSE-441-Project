import cv2
import numpy as np

# Load the MobileNet-SSD model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                               "MobileNetSSD_deploy.caffemodel")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Load video
cap = cv2.VideoCapture("cctv1.mp4")

# Define Region of Interest (ROI) rectangle
# Format: (x1, y1), (x2, y2)
roi_start = (0, 0)
roi_end = (450, 400)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Preprocess for detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    head_count_in_roi = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])

        if CLASSES[idx] == "person" and confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Estimate head region (upper 25%)
            head_bottom = startY + int((endY - startY) * 0.25)
            head_rect = (startX, startY, endX, head_bottom)

            # Check if the head centroid is inside ROI
            cx = int((startX + endX) / 2)
            cy = int((startY + head_bottom) / 2)

            if (roi_start[0] <= cx <= roi_end[0]) and (roi_start[1] <= cy <= roi_end[1]):
                head_count_in_roi += 1
                cv2.rectangle(frame, (startX, startY), (endX, head_bottom), (0, 255, 0), 2)

    # Draw ROI rectangle
    cv2.rectangle(frame, roi_start, roi_end, (255, 0, 0), 2)
    cv2.putText(frame, "ROI", (roi_start[0], roi_start[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show head count in ROI
    cv2.putText(frame, f"Heads in ROI: {head_count_in_roi}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Head Count in ROI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
