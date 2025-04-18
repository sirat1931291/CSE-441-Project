import cv2

# Load the model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                               "MobileNetSSD_deploy.caffemodel")

# List of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Load video file
cap = cv2.VideoCapture("cctv1.mp4")  # replace with your video file name

total_frames = 0
total_detected_heads = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    total_frames += 1

    # Prepare the frame for detection
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    frame_heads = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])

        # Check if the detected object is a person with enough confidence
        if CLASSES[idx] == "person" and confidence > 0.5:
            frame_heads += 1
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the estimated head region (top part of body box)
            head_height = startY + int((endY - startY) * 0.25)
            cv2.rectangle(frame, (startX, startY), (endX, head_height), (0, 255, 0), 2)

    total_detected_heads += frame_heads

    # Show the current frame with head boxes
    cv2.putText(frame, f"Heads: {frame_heads}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Head Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Total video frames processed:", total_frames)
print("Total head detections (non-unique):", total_detected_heads)
