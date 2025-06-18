def run(frame_queue, running):  ### CHANGED

    from deepface import DeepFace
    import cv2
    import os
    from scipy.spatial.distance import cosine
    import numpy as np

    # ======== CONFIG ========
    face_db_path = "../face_db"
    model_name = "VGG-Face"
    threshold = 0.4
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    last_label = None
    last_distance = 1.0
    smoothing_decay = 0.1

    # ======== Load face database ========
    print("[INFO] Loading face embeddings from DB...")
    database = {}

    for person_folder in os.listdir(face_db_path):
        person_path = os.path.join(face_db_path, person_folder)

        if not os.path.isdir(person_path):
            continue

        for img_file in os.listdir(person_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"[SKIPPED] {img_file} is not an image")
                continue

            try:
                img_path = os.path.join(person_path, img_file)
                rep = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=False)[0]["embedding"]

                if person_folder not in database:
                    database[person_folder] = []

                database[person_folder].append(rep)
                print(f"[LOADED] {person_folder}/{img_file}")

            except Exception as e:
                print(f"[WARNING] Skipped {img_file} due to error: {e}")

    if not database:
        print("[FATAL] No valid face data found in face_db.")
        return  ### CHANGED

    print("[INFO] FaceRecognitionPersonal process ready.")

    while running.value:  ### CHANGED
        if frame_queue.empty():
            continue

        frame = frame_queue.get()

        try:
            face_objs = DeepFace.extract_faces(frame, enforce_detection=False)
            for face in face_objs:
                try:
                    x = face['facial_area']['x']
                    y = face['facial_area']['y']
                    w = face['facial_area']['w']
                    h = face['facial_area']['h']
                except Exception as e:
                    print(f"[SKIP FACE] Bad bounding box: {e}")
                    continue

                cropped_face = face["face"]
                emb = DeepFace.represent(cropped_face, model_name=model_name, enforce_detection=False)[0]["embedding"]

                best_match = "Unknown"
                best_dist = 1.0

                for name, embeddings in database.items():
                    for ref_emb in embeddings:
                        dist = cosine(emb, ref_emb)
                        if dist < threshold and dist < best_dist:
                            best_match = f"Detected: {name}"
                            best_dist = dist

                # Flicker smoothing
                if best_dist < last_distance:
                    last_label = best_match
                    last_distance = best_dist
                else:
                    last_distance += smoothing_decay
                    if last_distance > threshold:
                        last_label = "Detecting..."

                # Draw bounding box and label
                if last_label == "Unknown":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(frame, last_label, (x, y - 10), label_font, 0.8, (255, 255, 255), 2)

        except Exception as e:
            print(f"[ERROR]: {e}")

        cv2.imshow("Face Recognition - DeepFace", frame)  ### CHANGED
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()  ### ADDED
