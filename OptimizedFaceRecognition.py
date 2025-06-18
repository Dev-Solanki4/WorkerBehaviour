import cv2
import numpy as np
import os
from deepface import DeepFace
import mediapipe as mp

# Configuration
MODEL_NAME = 'VGG-Face'
FACE_DB_PATH = 'face_db/'
FRAME_SKIP = 5
DETECTION_CONFIDENCE_THRESHOLD = 0.9
CROP_SIZE_THRESHOLD = 50

# Load face database
print("🔍 Loading face database...")
face_db = []
for folder in os.listdir(FACE_DB_PATH):
    person_path = os.path.join(FACE_DB_PATH, folder)
    if os.path.isdir(person_path):
        for file in os.listdir(person_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(person_path, file)
                try:
                    face_db.append((folder, image_path))
                except Exception as e:
                    print(f"❌ Failed to load {file}: {e}")
print(f"✅ Total images loaded: {len(face_db)}")

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# Start webcam
cap = cv2.VideoCapture(0)
frame_count = 0

def is_valid_crop(crop):
    return crop.shape[0] >= CROP_SIZE_THRESHOLD and crop.shape[1] >= CROP_SIZE_THRESHOLD

def find_identity(face_crop):
    try:
        for name, db_image_path in face_db:
            result = DeepFace.verify(
                face_crop, db_image_path, model_name=MODEL_NAME, enforce_detection=False
            )
            if result["verified"]:
                return name
    except Exception as e:
        print(f"⚠️ Verification error: {e}")
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    identity = None
    if results.detections:
        for detection in results.detections:
            if detection.score[0] < DETECTION_CONFIDENCE_THRESHOLD:
                continue

            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if not is_valid_crop(face_crop):
                continue

            identity = find_identity(face_crop)
            if identity:
                cv2.putText(frame, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Accurate Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
