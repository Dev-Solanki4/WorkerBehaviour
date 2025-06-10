from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 pose estimation model
model = YOLO("yolov8n-pose.pt")  # Use 'yolov8m-pose.pt' for better accuracy

def is_peace_sign(keypoints):
    """
    Check if the peace sign is shown by detecting if only index and middle fingers are extended.
    """ 
    try:
        fingers = {
            "index": (8, 6),
            "middle": (12, 10),
            "ring": (16, 14),
            "pinky": (20, 18)
        }

        extended = []
        for name, (tip, pip) in fingers.items():
            if tip >= len(keypoints) or pip >= len(keypoints):
                continue
            tip_y = keypoints[tip][1]
            pip_y = keypoints[pip][1]
            if tip_y < pip_y:  # Finger is extended
                extended.append(name)

        return set(extended) == {"index", "middle"}

    except:
        return False

# Start the webcam
cap = cv2.VideoCapture(0)

person_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run the model
    results = model.predict(frame, conf=0.5, verbose=False)
    annotated = frame.copy()
    person_count = 0

    # Loop through each detection
    for box, conf, keypoints in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].keypoints.xy):
        person_count += 1
        kp = keypoints.cpu().numpy()
        box = box.cpu().numpy().astype(int)
        conf = float(conf.cpu().numpy())

        # Draw person bounding box and label
        cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
        cv2.putText(annotated, f"Person #{person_count} ({conf:.2f})", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Check if peace sign is shown
        if kp.shape[0] >= 21 and is_peace_sign(kp):
            cv2.putText(annotated, f"Peace Sign ✌️ (#{person_count})", (box[0], box[3] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Highlight index and middle finger tips
            for i in [8, 12]:  # index and middle finger
                x, y = int(kp[i][0]), int(kp[i][1])
                cv2.circle(annotated, (x, y), 10, (0, 255, 0), -1)

        # Draw all other keypoints
        for x, y in kp:
            cv2.circle(annotated, (int(x), int(y)), 3, (255, 0, 0), -1)

    # Show output
    cv2.imshow("YOLOv8 Peace Sign Detector", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
