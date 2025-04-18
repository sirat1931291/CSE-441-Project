import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from collections import OrderedDict
import numpy as np
import time

# Load the pretrained model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                               "MobileNetSSD_deploy.caffemodel")

# Define the class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Initialize video stream
cap = cv2.VideoCapture("cctv1.mp4")

# For tracking unique heads
next_id = 0
tracked_heads = OrderedDict()

def get_centroid(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def is_close(c1, c2, thresh=50):
    return abs(c1[0] - c2[0]) < thresh and abs(c1[1] - c2[1]) < thresh

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # Create input blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    current_centroids = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])

        if CLASSES[idx] == "person" and confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Estimate head region (upper 25% of bounding box)
            headY = startY + int((endY - startY) * 0.25)
            centroid = get_centroid(startX, startY, endX, headY)
            current_centroids.append(centroid)

            cv2.rectangle(frame, (startX, startY), (endX, headY), (0, 255, 0), 2)

    # Update tracked_heads
    for centroid in current_centroids:
        found = False
        for track_id, track_centroid in tracked_heads.items():
            if is_close(centroid, track_centroid):
                tracked_heads[track_id] = centroid
                found = True
                break
        if not found:
            tracked_heads[next_id] = centroid
            next_id += 1

    # Show count on frame
    cv2.putText(frame, f"Unique Heads: {len(tracked_heads)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Unique Head Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
