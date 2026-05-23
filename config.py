"""
Конфигурация режимов работы.

DEBUG  = True  → CUDA + сохранение видео в output/
DEBUG  = False → CPU + только отображение
"""
import os

DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
SAVE_VIDEO = os.getenv('SAVE_VIDEO', 'True').lower() == 'true'
# ── Пути ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(PROJECT_ROOT, 'data', 'test_video.mp4')
if DEBUG:
    MODEL_PATH = 'yolo26n.pt'
else:
    MODEL_PATH = os.path.join(PROJECT_ROOT,'model', 'yolo26n.onnx')


OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
# Создаем output директорию
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ── Детектор ──
CONF_THRESH = 0.15

# ── Трекер ──
HIGH_THRESH = 0.5
LOW_THRESH = 0.2
IOU_THRESH = 0.3
MAX_AGE = 15
MIN_HITS = 8

# ── Классы ──
CLASS_NAMES = {2: "car", 3: "moto", 5: "bus", 7: "truck"}
