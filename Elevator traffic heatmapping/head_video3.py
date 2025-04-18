import cv2
import numpy as np

# Load the MobileNet-SSD model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                               "MobileNetSSD_deploy.caffemodel")

# Classes that the model can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Load your video file
cap = cv2.VideoCapture("cctv1.mp4")  # Replace with your video file name

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Get width and height of the frame
    (h, w) = frame.shape[:2]

    # Preprocess the frame for the network
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    head_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])

        # We're only interested in detecting people with confidence > 0.5
        if CLASSES[idx] == "person" and confidence > 0.5:
            head_count += 1

            # Get bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Define head region (top part of bounding box)
            head_bottom = startY + int((endY - startY) * 0.25)
            cv2.rectangle(frame, (startX, startY), (endX, head_bottom), (0, 255, 0), 2)

    # Show count of heads in the current frame
    cv2.putText(frame, f"Heads in Frame: {head_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the video with head count
    cv2.imshow("Head Counter", frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
