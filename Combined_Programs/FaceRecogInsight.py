def run(frame_queue, running):
    """
    Face‑recognition worker using InsightFace (buffalo_l) on GPU.

    Parameters
    ----------
    frame_queue : multiprocessing.Queue
        Queue that receives BGR frames (np.ndarray) from the camera process.
    running : multiprocessing.Value (c_bool)
        Shared flag; keep looping while running.value is True.
    """
    # ────────────────────────────────────────────────────────────── imports
    import os, time, cv2, torch
    from insightface.app import FaceAnalysis
    import numpy as np

    # ─────────────────────────────────────────────── configuration
    face_db_path        = "../face_db"     #  ← adjust if needed
    similarity_threshold = 0.60            # cosine similarity threshold
    smoothing_decay      = 0.03            # controls flicker‑free label hold
    label_font           = cv2.FONT_HERSHEY_SIMPLEX
    display_window_name  = "Face Recognition - InsightFace"

    # ───────────────────────────────────────────── prepare model GPU
    print("[INFO] Initialising InsightFace (buffalo_l) on GPU …")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)                  # 0 = first CUDA device

    # ────────────────────────────────────── build the face database
    print("[INFO] Building face‑embedding database …")
    face_db = {}                           # {person : avg_embedding (Tensor)}

    for person in os.listdir(face_db_path):
        person_dir = os.path.join(face_db_path, person)
        if not os.path.isdir(person_dir):
            continue

        embeds = []
        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(person_dir, img_file)
            img      = cv2.imread(img_path)
            if img is None:
                print(f"[SKIP] Couldn’t read {img_path}")
                continue
            img = cv2.resize(img, (640, 480))      # keep resolution consistent
            faces = app.get(img)
            if faces:
                embeds.append(torch.tensor(faces[0].embedding, device="cuda"))

        if embeds:
            face_db[person] = torch.stack(embeds).mean(dim=0)
            print(f"[LOADED] {person:15} {len(embeds)} images")
        else:
            print(f"[WARN ] No usable faces for {person}")

    if not face_db:
        print("[FATAL] face_db empty — aborting worker.")
        return

    # ───────────────────────────────────────── tracking variables
    last_label     = "Detecting…"
    last_conf      = 0.0                   # highest similarity in previous frame
    t_last_frame   = time.time()

    # ────────────────────────────────────────────── processing loop
    print("[INFO] Face‑recognition worker ready.")
    cv2.namedWindow(display_window_name, cv2.WINDOW_NORMAL)

    while running.value:
        if frame_queue.empty():
            # small sleep avoids busy‑wait when queue starves
            time.sleep(0.002)
            continue

        frame = frame_queue.get()
        frame = cv2.resize(frame, (640, 480))      # match training resolution

        faces = app.get(frame)
        for face in faces:
            emb   = torch.tensor(face.embedding, device="cuda")
            name  = "Unknown"
            best  = 0.0

            # compare against DB (all on GPU, very fast)
            for person, db_emb in face_db.items():
                sim = torch.nn.functional.cosine_similarity(emb, db_emb, dim=0).item()
                if sim > similarity_threshold and sim > best:
                    name, best = person, sim

            # ───────── simple temporal smoothing to reduce flicker
            if best >= last_conf:
                last_label, last_conf = name, best
            else:
                last_conf = max(0.0, last_conf - smoothing_decay)
                if last_conf < similarity_threshold:
                    last_label = "Detecting…"

            # ───────── draw UI
            x1, y1, x2, y2 = map(int, face.bbox)   # bounding box
            color = (0, 255, 0) if last_label != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, last_label, (x1, y1 - 10),
                        label_font, 0.75, (255, 255, 255), 2)

        # FPS counter (optional)
        now  = time.time()
        fps  = 1.0 / (now - t_last_frame)
        t_last_frame = now
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                    label_font, 0.7, (255, 255, 0), 2)

        # show frame
        cv2.imshow(display_window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:        # ESC pressed
            running.value = False
            break

    cv2.destroyAllWindows()
    print("[INFO] Face‑recognition worker stopped.")
