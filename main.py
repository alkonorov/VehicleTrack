import cv2 as cv
import time
import numpy as np
from video import VideoStream
from detector import VehicleDetector
from tracker import ByteTracker
from counter import VehicleCounter
from line_selector import LineSelector
from config import *
# VIDEO_PATH = '/home/alkon/code/test_tracker/data/test_video.mp4'
# MODEL_PATH = '/home/alkon/code/test_tracker/yolo26n.onnx'
# CLASS_NAMES = {2: "car", 3: "moto", 5: "bus", 7: "truck"}

# ── Инициализация ──
video = VideoStream(VIDEO_PATH)
if DEBUG:
    detector = VehicleDetector(MODEL_PATH,device='cuda', conf_thresh=0.15)
tracker = ByteTracker(
    hight_thresh=0.5,
    low_thresh=0.2,
    iou_thresh=0.3,
    max_age=10,
    min_hits=7
)

if not video.open_video():
    exit()

# ── Выбор линии подсчёта ──
ret, first_frame = video.read()
if not ret:
    print("Не удалось прочитать первый кадр")
    exit()

# selector = LineSelector()
# line = selector.select(first_frame)
#
# if line is None:
#     print("Линия не выбрана, выход")
#     video.release()
#     exit()
#
# line_start, line_end = line

# Создаём счётчик
# counter = VehicleCounter(line_start, line_end, CLASS_NAMES)
counter = VehicleCounter(
    direction_axis='diagonal',
    min_age=20,
    min_displacement=70,
    direction_consistency=0.7,
    min_movement_frames=7
)
# ── Перемотка на начало ──
video.release()
video.open_video()



####TEST отладка########
writer = None
if DEBUG:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "result.mp4")
    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    # Приводим размеры к целым числам
    width = int(video.width)
    height = int(video.height)

    writer = cv.VideoWriter(
        output_path,
        fourcc,
        float(video.video_fps),  # fps тоже лучше привести к float
        (width, height)
    )

    if writer.isOpened():
        print(f"✅ Запись в: {output_path}")
        print(f"   Размер: {width}x{height}, FPS: {video.video_fps}")
    else:
        print("⚠️ Не удалось создать VideoWriter")
        writer = None



#####################

# ── Цвета для треков ──
def get_color(track_id):
    np.random.seed(track_id * 77)
    return tuple(int(c) for c in np.random.randint(50, 255, 3))


# ── Основной цикл ──
prev_time = time.time()
fps_counter = 0
display_fps = 0.0

while True:
    ret, frame = video.read()
    if not ret:
        break

    # 1. Детекция
    detections = detector.detect(frame)

    # 2. Трекинг
    tracks = tracker.update(detections)

    # 3. Подсчёт пересечений
    crossed_ids = counter.update(tracks)

    # 4. Визуализация

    # Линия подсчётаq
    # cv.line(frame, line_start, line_end, (0, 0, 255), 3)
    # cv.putText(frame, "COUNTING LINE",
    #            (line_start[0], line_start[1] - 15),
    #            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Треки
    for track in tracks:
        x1, y1, x2, y2 = track.bbox_xyxy.astype(int)
        tid = track.track_id

        # Подсветка при пересечении линии
        if tid in crossed_ids:
            color = (0, 255, 255)  # жёлтая вспышка
            thickness = 4
        elif tid in counter.counted_ids:
            color = (0, 255, 0)  # зелёный = уже посчитан
            thickness = 2
        else:
            color = (105, 105, 105)#get_color(tid)
            thickness = 2

        cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # class_name = CLASS_NAMES.get(track.class_id, "?")
        label = f"ID:{tid}" # {class_name}"
        cv.putText(frame, label, (x1, y1 - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Траектория
        if len(track.history) > 1:
            pts = np.array(list(track.history), dtype=np.int32)
            cv.polylines(frame, [pts], isClosed=False, color=color, thickness=2)

    # Панель статистики
    stats_lines = counter.get_display_text()
    panel_h = 30 + len(stats_lines) * 25
    overlay = frame.copy()
    cv.rectangle(overlay, (10, 50), (350, 50 + panel_h), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, text in enumerate(stats_lines):
        cv.putText(frame, text, (20, 75 + i * 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # FPS
    fps_counter += 1
    elapsed = time.time() - prev_time
    if elapsed >= 1.0:
        display_fps = fps_counter / elapsed
        fps_counter = 0
        prev_time = time.time()
        # print(f"Кадр: {video.frame_count}/{video.total_frames} | "
        #       f"Треков: {len(tracks)} | FPS: {display_fps:.1f} | "
        #       f"Counted: {counter.total_up + counter.total_down}")
        stats = counter.get_stats()
        print(f"Кадр: {video.frame_count}/{video.total_frames} | "
              f"Треков: {len(tracks)} | FPS: {display_fps:.1f} | "
              f"Counted: {stats['total']}")

    cv.putText(frame, f"FPS: {display_fps:.1f}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if writer:
        writer.write(frame)
    cv.imshow("Vehicle Counter", frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break

if writer:
    writer.release()
#Итоговая статистика
stats = counter.get_stats()
print(f"Всего транспорта: {stats['total']}")
# print(f"  ↑ Вверх:  {stats['total_up']}")
# print(f"  ↓ Вниз:   {stats['total_down']}")
print(f"  Направление A: {stats['direction_a']}")
print(f"  Направление B: {stats['direction_b']}")


video.release()
cv.destroyAllWindows()
