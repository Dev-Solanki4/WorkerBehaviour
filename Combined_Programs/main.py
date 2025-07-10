import multiprocessing
import cv2
import time

import FaceRecognitionPersonal
import FaceRecogInsight
import MultiPoseEstimation
import AdvancePoseEstimation
import StickFigure

def frame_producer(frame_queue, running):
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(r"C:\Users\dev.solanki\OneDrive - HRPL RESTAURANTS PRIVATE LIMITED\Desktop\CON_CUP_LINE-1_Hocco_Hocco_20250620090000_20250620091058_135481.mp4")

    # if not cap.isOpened():
    #     print("[ERROR] Cannot access the webcam")
    #     running.value = False
    #     return

    # print("[INFO] Webcam started. Press ESC to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        # Avoid queue overflow
        if not frame_queue.full():
            frame_queue.put(frame)

        # Optional: display what main sees
        cv2.imshow("Main - Shared Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            running.value = False
            break

    cap.release()
    cv2.destroyAllWindows()
    running.value = False

if __name__ == "__main__":
    # Shared objects
    frame_queue = multiprocessing.Queue(maxsize=5)  # shared feed
    running = multiprocessing.Value('b', True)  # shared flag

    # Start consumers
    p1 = multiprocessing.Process(target=FaceRecogInsight.run, args=(frame_queue, running))
    p2 = multiprocessing.Process(target=MultiPoseEstimation.run, args=(frame_queue, running))
    p3 = multiprocessing.Process(target=StickFigure.run, args=(frame_queue, running))

    # Start producer
    producer = multiprocessing.Process(target=frame_producer, args=(frame_queue, running))

    # Start all
    producer.start()
    p1.start()
    p2.start()
    p3.start()

    # Wait for all
    producer.join()
    p1.join()
    p2.join()
    p3.join()

    print("[INFO] All processes finished.")
