from insightface.app import FaceAnalysis
import os
import cv2
import numpy as np
import time
import torch

# Use a smaller model for faster inference
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)  # GPU mode

pTime = 0
face_db = {}

# Load face database and compute average embeddings
for person in os.listdir("face_db"):
    person_path = os.path.join("face_db", person)
    embeddings = []

    if os.path.isdir(person_path):
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (640, 480))  # Resize for consistency
            faces = app.get(img)
            if faces:
                embeddings.append(torch.tensor(faces[0].embedding).cuda())

        if embeddings:
            avg_embedding = torch.stack(embeddings).mean(dim=0)
            face_db[person] = avg_embedding

# Cosine similarity using GPU
def cosine_similarity_gpu(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

# Use DirectShow backend for faster capture (Windows only)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))  # Resize to reduce load
    faces = app.get(frame)

    for face in faces:
        emb = torch.tensor(face.embedding).cuda()
        name = "Unknown"
        max_sim = 0.0

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        for person, db_emb in face_db.items():
            sim = cosine_similarity_gpu(emb, db_emb)
            if sim > 0.6 and sim > max_sim:
                name = person
                max_sim = sim

        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"FPS : {fps:.2f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"{name}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("InsightFace Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
