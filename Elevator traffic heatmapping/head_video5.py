import cv2
import numpy as np

# Load MobileNet-SSD model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                               "MobileNetSSD_deploy.caffemodel")

# Class labels MobileNet SSD was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Load video
cap = cv2.VideoCapture("cctv1.mp4")

# Define ROI area (change as needed)
roi_start = (0, 0)
roi_end = (450, 400)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optionally resize the frame slightly larger (helps with far objects)
    frame = cv2.resize(frame, (800, 600))

    (h, w) = frame.shape[:2]

    # Prepare blob for detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    head_count_in_roi = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])

        # Lower confidence threshold for better detection of small heads
        if CLASSES[idx] == "person" and confidence > 0.3:
            # Get person box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Estimate head region as top 25% of person bounding box
            head_bottom = startY + int((endY - startY) * 0.25)
            cx = int((startX + endX) / 2)
            cy = int((startY + head_bottom) / 2)

            # Check if head centroid is in ROI
            if (roi_start[0] <= cx <= roi_end[0]) and (roi_start[1] <= cy <= roi_end[1]):
                head_count_in_roi += 1
                # Draw head region
                cv2.rectangle(frame, (startX, startY), (endX, head_bottom), (0, 255, 0), 2)

            # Draw full person box for reference (optional)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (100, 100, 255), 1)

    # Draw ROI area
    cv2.rectangle(frame, roi_start, roi_end, (255, 0, 0), 2)
    cv2.putText(frame, "ROI", (roi_start[0], roi_start[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show head count in ROI
    cv2.putText(frame, f"Heads in ROI: {head_count_in_roi}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display result
    cv2.imshow("Improved Head Detection in ROI", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
