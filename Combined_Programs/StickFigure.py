def run(frame_queue, running):
    """
    Multi‑person body‑pose (YOLOv8‑Pose) + hands (MediaPipe) visualiser.
    """

    # ── imports ───────────────────────────────────────────────────────
    import cv2, time, torch
    from collections import deque, defaultdict
    from ultralytics import YOLO
    import mediapipe as mp

    # ── models ────────────────────────────────────────────────────────
    pose_model = YOLO("yolov8m-pose.pt").cuda()     # n/m/l‑pose
    pose_model.fuse()

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    hands    = mp_hands.Hands(False, max_num_hands=4,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)

    # ── skeleton edge lists ───────────────────────────────────────────
    EDGES_17 = [  # COCO (YOLO) order
        (5,7), (7,9), (6,8), (8,10),
        (5,6), (5,11), (6,12), (11,12),
        (11,13), (13,15), (12,14), (14,16),
        (0,1), (0,2), (1,3), (2,4)
    ]
    EDGES_33 = [  # MediaPipe full‑body order
        (11,13),(13,15),(12,14),(14,16),(11,12),(11,23),(12,24),(23,24),
        (23,25),(25,27),(24,26),(26,28),(27,31),(28,32),
        (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8)
    ]

    def draw_pose(img, kps, color=(0,255,0), thick=2):
        """Draw stick‑figure; works for 17‑ or 33‑kp layouts."""
        h, w, _ = img.shape
        edges = EDGES_33 if len(kps) >= 33 else EDGES_17
        for s, e in edges:
            if s >= len(kps) or e >= len(kps):
                continue
            xs, ys = int(kps[s,0]*w), int(kps[s,1]*h)
            xe, ye = int(kps[e,0]*w), int(kps[e,1]*h)
            cv2.line(img, (xs,ys), (xe,ye), color, thick)

    # ── helpers ───────────────────────────────────────────────────────
    FONT, SCALE, THK = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    t0 = time.time()

    print("[INFO] StickFigure worker ready.")
    while running.value:
        if frame_queue.empty():
            time.sleep(0.002); continue

        frame = frame_queue.get()
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── pose + tracker ────────────────────────────────────────────
        res = pose_model.track(
                frame, imgsz=960, conf=0.4,
                tracker="botsort.yaml", persist=True, verbose=False)[0]

        out = frame.copy()

        if res.keypoints is not None:
            for kp, box in zip(res.keypoints, res.boxes):
                tid = int(box.id.item()) if box.id is not None else -1
                if tid == -1:
                    continue

                kps = kp.xy[0].cpu().numpy()           # pixel coords
                kps[:,0] /= frame.shape[1]; kps[:,1] /= frame.shape[0]
                draw_pose(out, kps)

                x1, y1, _, _ = map(int, box.xyxy[0])
                cv2.putText(out, f"ID {tid}", (x1, y1-8),
                            FONT, SCALE, (255,255,255), THK+1)

        # ── hands overlay ─────────────────────────────────────────────
        hand_det = hands.process(rgb)
        if hand_det.multi_hand_landmarks:
            for h_lm in hand_det.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    out, h_lm, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec((255,0,0), 1, 2),
                    mp_draw.DrawingSpec((0,255,255), 1, 1))

        # FPS read‑out
        fps = 1.0 / (time.time() - t0); t0 = time.time()
        cv2.putText(out, f"FPS {fps:.1f}", (8,24),
                    FONT, 0.6, (0,255,0), 2)

        cv2.imshow("Multi‑worker Pose + Hands", out)
        if cv2.waitKey(1) & 0xFF == 27:        # ESC
            running.value = False
            break

    cv2.destroyAllWindows()
