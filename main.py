import cv2 as cv
import time, os
from video import VideoStream
from detector import Detection, VehicleDetector
from tracker import ByteTracker

VIDEO_PATH = '/home/alkon/code/test_tracker/data/test_video.mp4'
MODEL_PATH = '/home/alkon/code/test_tracker/yolo26n.onnx'
CLASS_NAMES = {2: "car", 3: "moto", 5: "bus", 7: "truck"}

video = VideoStream(VIDEO_PATH)
detector = VehicleDetector(MODEL_PATH)
tracker = ByteTracker(
    hight_thresh=0.5,
    low_thresh=0.1,
    iou_thresh=0.3,
    max_age=30,
    min_hits=3
)

if not video.open_video():
    exit()

prev_time = time.time()
fps_counter = 0
display_fps = 0.0


import numpy as np
def get_color(track_id):
    """Стабильный цвет для каждого ID"""
    np.random.seed(track_id * 77)
    return tuple(int(c) for c in np.random.randint(50, 255, 3))


while True:
    ret, frame = video.read()
    if not ret:
        break

    detections = detector.detect(frame)

    tracks = tracker.update(detections)

    fps_counter += 1
    elapsed = time.time() - prev_time
    if elapsed >= 1.0:
        display_fps = fps_counter / elapsed
        fps_counter = 0
        prev_time = time.time()
        print(f"обработка кадров: {video.frame_count} из {video.total_frames}")
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy.astype(int)
        cv.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        # class_name = CLASS_NAMES[detection.class_id]
        # cv.putText(frame,class_name,(x1, y1 - 5),
        #                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv.putText(frame, f"FPS:{display_fps:.2f}",(10,30), cv.FONT_HERSHEY_PLAIN,1,
               (0,255,0),2  )
    cv.imshow("video", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv.destroyAllWindows()