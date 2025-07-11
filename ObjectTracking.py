import os
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import mediapipe as mp 

cap = cv2.VideoCapture(0)
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2)
cap.set(3,1280)
cap.set(4,720)

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands(max_num_hands=1)

duration = 0
detection_ongoing = False
detection_start_time = 0

while True:
    succ, img = cap.read()
    if not succ:
        break

    results = model(img)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hand = hands.process(imgRGB)

    detections = []
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append(([x1, y1, x2, y2], conf, 'person'))

    hand_detected = hand.multi_hand_landmarks is not None
    current_time = time.time()

    if hand_detected:
        if not detection_ongoing:
            detection_start_time = current_time
            detection_ongoing = True
    else:
        if detection_ongoing:
            duration += current_time - detection_start_time
            detection_ongoing = False

    if detection_ongoing:
        live_duration = current_time - detection_start_time
    else:
        live_duration = 0

    total_time = int(duration + live_duration)

    tracks = tracker.update_tracks(detections, frame=img)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(img, f"Hand Time: {total_time}s", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if hand.multi_hand_landmarks:
        for handlms in hand.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handlms)

    cv2.imshow("Object Tracking", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

if detection_ongoing:
    duration += time.time() - detection_start_time

cap.release()
cv2.destroyAllWindows()
