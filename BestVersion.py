import cv2
import os
import time
import numpy as np
from collections import deque
import statistics
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp

# ------------------ Facial Recognition ------------------ #
def load_face_database(app, db_path="face_db"):
    face_db = {}
    for person in os.listdir(db_path):
        person_path = os.path.join(db_path, person)
        embeddings = []
        if os.path.isdir(person_path):
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path)
                faces = app.get(img)
                if faces:
                    embeddings.append(faces[0].embedding)
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                face_db[person] = avg_embedding
    return face_db

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_faces(app, frame, face_db):
    faces = app.get(frame)
    recognized = []
    for face in faces:
        emb = face.embedding
        name = "Unknown"
        max_sim = 0.0
        for person, db_emb in face_db.items():
            sim = cosine_similarity(emb, db_emb)
            if sim > 0.6 and sim > max_sim:
                name = person
                max_sim = sim
        recognized.append((face.bbox.astype(int), name))
    return recognized

# ------------------ Pose Estimation ------------------ #
def estimate_pose(image, pose, prev_left_hand, prev_right_hand, activity_queue):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = pose.process(image_rgb)
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    activity = "Unknown"
    working = False
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        hip_knee_diff = abs(left_hip.y - left_knee.y)
        shoulder_hip_diff = abs(left_shoulder.y - left_hip.y)
        hand_diff = abs(left_hand.x - right_hand.x)
        shoulder_knee_diff = abs(left_shoulder.y - left_knee.y)
        pose_ratio = hip_knee_diff / (shoulder_hip_diff + 1e-5)

        if shoulder_knee_diff < 0.2:
            posture = "Crouching"
            idle = False
        elif pose_ratio < 0.6 and hand_diff < 0.1:
            posture = "Sitting"
            idle = True
        elif pose_ratio < 0.6:
            posture = "Sitting"
            idle = False
        elif hand_diff < 0.1 and shoulder_hip_diff < 0.65:
            posture = "Standing"
            idle = True
        else:
            posture = "Standing"
            idle = False

        if prev_left_hand and prev_right_hand:
            left_movement = abs(left_hand.x - prev_left_hand[0]) + abs(left_hand.y - prev_left_hand[1])
            right_movement = abs(right_hand.x - prev_right_hand[0]) + abs(right_hand.y - prev_right_hand[1])
            if left_movement > 0.02 or right_movement > 0.02:
                working = True

        prev_left_hand = (left_hand.x, left_hand.y)
        prev_right_hand = (right_hand.x, right_hand.y)

        if idle and not working:
            activity = f"{posture} + Idle"
        elif working:
            activity = f"{posture} + Working"
        else:
            activity = posture

        activity_queue.append(activity)
        stable_activity = statistics.mode(activity_queue)
    else:
        stable_activity = "No Person Detected"

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return image, stable_activity, prev_left_hand, prev_right_hand, working

# ------------------ Hand Tracking ------------------ #
def track_hands(img, model, tracker, hands, duration, detection_ongoing, detection_start_time, working):
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

    # hand_detected = hand.multi_hand_landmarks is not None
    current_time = time.time()

    if working:
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
            mp.solutions.drawing_utils.draw_landmarks(img, handlms)

    return img, duration, detection_ongoing, detection_start_time

# ------------------ Main Loop ------------------ #
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0)
    face_db = load_face_database(app)

    model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2)
    hands = mp.solutions.hands.Hands(max_num_hands=1)
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    activity_queue = deque(maxlen=10)
    prev_left_hand = None
    prev_right_hand = None
    duration = 0
    detection_ongoing = False
    detection_start_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        recognized_faces = recognize_faces(app, frame, face_db)
        for bbox, name in recognized_faces:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
            cv2.putText(frame, f"{name}", (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        known_detected = any(name != "Unknown" for _, name in recognized_faces)

        if known_detected:
            frame, activity, prev_left_hand, prev_right_hand, working_detected = estimate_pose(
                frame, pose, prev_left_hand, prev_right_hand, activity_queue)
            cv2.putText(frame, activity, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            if "Working" in activity:
                frame, duration, detection_ongoing, detection_start_time = track_hands(
                    frame, model, tracker, hands, duration, detection_ongoing, detection_start_time, working_detected)

        cv2.imshow("Integrated Worker Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if detection_ongoing:
        duration += time.time() - detection_start_time

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
