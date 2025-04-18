import cv2
import numpy as np

# Load the MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                               "MobileNetSSD_deploy.caffemodel")

# List of class labels MobileNet SSD was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Load video
cap = cv2.VideoCapture("cctv1.mp4")  # Replace with your video file

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

# Define Region of Interest (ROI)
roi_start = (0, 0)
roi_end = (450, 400)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: Resize for better detection
    frame = cv2.resize(frame, (800, 600))
    (h, w) = frame.shape[:2]

    # Prepare the image blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    fullbody_count_in_roi = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])

        if CLASSES[idx] == "person" and confidence > 0.3:
            # Get full-body bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Calculate center of full body box
            cx = int((startX + endX) / 2)
            cy = int((startY + endY) / 2)

            # Check if center is within ROI
            if (roi_start[0] <= cx <= roi_end[0]) and (roi_start[1] <= cy <= roi_end[1]):
                fullbody_count_in_roi += 1
                # Draw full-body bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Draw ROI box
    cv2.rectangle(frame, roi_start, roi_end, (255, 0, 0), 2)
    cv2.putText(frame, "ROI", (roi_start[0], roi_start[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display person count in ROI
    cv2.putText(frame, f"Persons in ROI: {fullbody_count_in_roi}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the video
    cv2.imshow("Full Body Detection in ROI", frame)

    # Exit on 'q'
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
