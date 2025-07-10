def run(frame_queue, running):

    import cv2
    import mediapipe as mp
    from collections import deque
    import statistics

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Variables for smoothing
    activity_queue = deque(maxlen=10)
    prev_left_hand = None
    prev_right_hand = None

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        print("[INFO] AdvancePoseEstimation process ready.")

        while running.value:  
            if frame_queue.empty():
                continue

            frame = frame_queue.get()

            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Pose detection
            results = pose.process(image)

            # Draw keypoints
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            activity = "Unknown"

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get required keypoints
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                # Distance-based features
                hip_knee_diff = abs(left_hip.y - left_knee.y)
                shoulder_hip_diff = abs(left_shoulder.y - left_hip.y)
                hand_diff = abs(left_hand.x - right_hand.x)
                shoulder_knee_diff = abs(left_shoulder.y - left_knee.y)

                # Ratio-based feature
                pose_ratio = hip_knee_diff / (shoulder_hip_diff + 1e-5)

                # Posture logic
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

                # Hand movement detection
                working = False
                if prev_left_hand and prev_right_hand:
                    left_movement = abs(left_hand.x - prev_left_hand[0]) + abs(left_hand.y - prev_left_hand[1])
                    right_movement = abs(right_hand.x - prev_right_hand[0]) + abs(right_hand.y - prev_right_hand[1])

                    if left_movement > 0.02 or right_movement > 0.02:
                        working = True

                # Update previous hand position
                prev_left_hand = (left_hand.x, left_hand.y)
                prev_right_hand = (right_hand.x, right_hand.y)

                # Combine logic
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

            # Display
            cv2.putText(image, stable_activity, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Worker Pose & Activity Detection', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cv2.destroyAllWindows()  
