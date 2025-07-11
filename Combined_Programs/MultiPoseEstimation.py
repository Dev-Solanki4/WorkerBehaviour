def run(frame_queue, running):
    import cv2, torch, statistics, time, math, sqlite3
    import numpy as np
    from collections import defaultdict, deque
    from ultralytics import YOLO

    model   = YOLO("yolov8m-pose.pt")
    tracker = "bytetrack.yaml"
    model.cuda().fuse()

    # ───── Database Setup ─────
    conn = sqlite3.connect("activity_log.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ActivityLog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            worker_id INTEGER,
            activity TEXT,
            camera_id TEXT
        )
    """)
    conn.commit()

    # ───── Per-ID state ─────
    STATE = defaultdict(lambda: {
        "buf": deque(maxlen=10),
        "prev_kps": None,
        "last_activity": None  # For detecting change
    })

    COCO = {
        "HEAD": 0,  "SH_L": 5,  "SH_R": 6,
        "EL_L": 7,  "EL_R": 8,  "WR_L": 9, "WR_R": 10,
        "HIP_L": 11, "HIP_R": 12,
        "KN_L": 13,  "KN_R": 14,
        "AN_L": 15,  "AN_R": 16
    }

    def angle3pts(a, b, c):
        ba, bc = a - b, c - b
        cosang = (ba @ bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return math.degrees(math.acos(max(-1, min(1, cosang))))

    def classify_pose(kps):
        y_head = kps[COCO["HEAD"], 1]
        y_heel = min(kps[COCO["AN_L"], 1], kps[COCO["AN_R"], 1])
        height = y_heel - y_head + 1e-6

        y_sh = (kps[COCO["SH_L"],1] + kps[COCO["SH_R"],1]) / 2
        y_kn = (kps[COCO["KN_L"],1] + kps[COCO["KN_R"],1]) / 2
        knee_sh_ratio = (y_kn - y_sh) / height

        ang_l = angle3pts(kps[COCO["HIP_L"]], kps[COCO["KN_L"]], kps[COCO["AN_L"]])
        ang_r = angle3pts(kps[COCO["HIP_R"]], kps[COCO["KN_R"]], kps[COCO["AN_R"]])
        knee_angle = min(ang_l, ang_r)

        if knee_angle < 70 and knee_sh_ratio < 0.40:
            posture = "Crouching"
        elif knee_sh_ratio < 0.55:
            posture = "Sitting"
        else:
            posture = "Standing"

        wrist_gap = abs(kps[COCO["WR_L"],0] - kps[COCO["WR_R"],0])
        idle = wrist_gap < 0.12

        return posture, idle

    def hand_motion(prev, now):
        return (
            abs(now[COCO["WR_L"],0] - prev[COCO["WR_L"],0]) +
            abs(now[COCO["WR_L"],1] - prev[COCO["WR_L"],1]) +
            abs(now[COCO["WR_R"],0] - prev[COCO["WR_R"],0]) +
            abs(now[COCO["WR_R"],1] - prev[COCO["WR_R"],1])
        )

    # ───── Main Loop ─────
    fps_t0 = time.time()
    while running.value:
        if frame_queue.empty():
            time.sleep(0.002)
            continue
        frame = frame_queue.get()

        res = model.track(frame, tracker=tracker, persist=True, verbose=False,
                          imgsz=960, conf=0.4)[0]

        annotated = res.plot(
            boxes=False, labels=False, probs=False,
            kpt_line=True, kpt_radius=2
        )

        if res.keypoints is not None:
            for kp, box in zip(res.keypoints, res.boxes):
                wid = int(box.id.item()) if box.id is not None else -1
                if wid == -1:
                    continue

                scale = torch.tensor(
                    [frame.shape[1], frame.shape[0]],
                    device=kp.xy.device, dtype=kp.xy.dtype
                )
                kps = (kp.xy[0] / scale).cpu().numpy()

                posture, idle = classify_pose(kps)

                state = STATE[wid]
                working = False
                if state["prev_kps"] is not None:
                    working = hand_motion(state["prev_kps"], kps) > 0.03
                state["prev_kps"] = kps

                state["buf"].append(
                    f"{posture} + {'Working' if working else 'Idle'}"
                    if (idle or working) else posture
                )
                activity = statistics.mode(state["buf"])

                # ───── Log if activity changed ─────
                if state["last_activity"] != activity:
                    cursor.execute("""
                        INSERT INTO ActivityLog (timestamp, worker_id, activity, camera_id)
                        VALUES (datetime('now'), ?, ?, ?)
                    """, (wid, activity, "CAM1"))  # Adjust camera_id if needed
                    conn.commit()
                    state["last_activity"] = activity

                # ───── Draw box & label ─────
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0,255,0) if "Working" in activity else (0,0,255)
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 1)
                cv2.putText(annotated, f"ID {wid}: {activity}",
                            (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255,255,255), 2)

        fps = 1 / (time.time() - fps_t0); fps_t0 = time.time()
        cv2.putText(annotated, f"FPS {fps:.1f}", (8,24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Worker Pose & Activity Detection", annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            running.value = False
            break

    conn.close()
    cv2.destroyAllWindows()
