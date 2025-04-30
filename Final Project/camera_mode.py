import cv2
import threading
import pyttsx3
from ultralytics import YOLO

engine = pyttsx3.init()
speak_lock = threading.Lock()

def speak_count(count):
    if speak_lock.locked():
        return
    with speak_lock:
        engine.say(f"{count} people")
        engine.runAndWait()

def run_video_mode(update_callback, stop_event):
    model = YOLO("yolo11s.pt")
    cap = cv2.VideoCapture(0)

    roi_top_left = (0, 0)
    roi_bottom_right = (800, 500)
    last_count = -1

    while cap.isOpened():
        if stop_event.is_set():
            break

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        person_count = 0

        cv2.rectangle(frame, roi_top_left, roi_bottom_right, (255, 0, 0), 2)

        for result in results:
            boxes = result.boxes
            for i in range(len(boxes.cls)):
                if int(boxes.cls[i]) == 0:
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if roi_top_left[0] <= cx <= roi_bottom_right[0] and roi_top_left[1] <= cy <= roi_bottom_right[1]:
                        person_count += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

        if person_count != last_count:
            last_count = person_count
            status = "Free" if person_count < 6 else "Medium" if person_count < 9 else "Busy"
            update_callback(f"{person_count} People - {status}")
            threading.Thread(target=speak_count, args=(person_count,), daemon=True).start()

        cv2.imshow("YOLOv11 - People in ROI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or stop_event.is_set():
            break

    cap.release()
    cv2.destroyAllWindows()
