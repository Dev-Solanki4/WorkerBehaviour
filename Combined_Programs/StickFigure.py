def run(frame_queue, running):  ### CHANGED

    import cv2
    import mediapipe as mp

    # Initialize MediaPipe modules
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose()
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

    # Pose keypoint connections (custom skeleton)
    POSE_CONNECTIONS = [
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 12),
        (11, 23), (12, 24),
        (23, 24),
        (23, 25), (25, 27),
        (24, 26), (26, 28),
        (27, 31), (28, 32),
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8)
    ]

    def draw_landmark_lines(image, landmarks, connections, color=(0, 255, 0)):
        h, w, _ = image.shape
        for start_idx, end_idx in connections:
            try:
                x1 = int(landmarks[start_idx].x * w)
                y1 = int(landmarks[start_idx].y * h)
                x2 = int(landmarks[end_idx].x * w)
                y2 = int(landmarks[end_idx].y * h)
                cv2.line(image, (x1, y1), (x2, y2), color, 2)
            except:
                continue

    print("[INFO] StickFigure process ready.")

    while running.value:  ### CHANGED
        if frame_queue.empty():
            continue

        frame = frame_queue.get()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_result = pose.process(rgb)
        hand_result = hands.process(rgb)

        # Draw pose
        if pose_result.pose_landmarks:
            draw_landmark_lines(frame, pose_result.pose_landmarks.landmark, POSE_CONNECTIONS, color=(0, 255, 0))

        # Draw hands
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )

        cv2.imshow("Stickman Pose + Hands", frame)  ### CHANGED
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()  ### CHANGED
