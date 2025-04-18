import cv2

# Load pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

# Class labels for COCO
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    # Create blob for the model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])

        if CLASSES[idx] == "person" and confidence > 0.5:
            count += 1
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            # Estimate head box (top 1/4 of the person box)
            headY = startY + int((endY - startY) * 0.25)
            cv2.rectangle(frame, (startX, startY), (endX, headY), (0, 255, 0), 2)

    # Display count
    cv2.putText(frame, f"Head Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    cv2.imshow("Head Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
