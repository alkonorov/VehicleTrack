from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    bbox_xyxy: np.ndarray # [x1,y1,x2,y2]
    score: float
    class_id: int



class VehicleDetector():
    def __init__(self, model_path, classes=None, device='cpu', conf_thresh=0.1):
        if classes is None:
            classes = [2, 3, 5, 7]
        self.device = device
        self.conf_thresh = conf_thresh
        self.model = YOLO(model_path)
        self.classes = classes # только ТС (COCO)

        self._warmup_model() # Прогрев модели

    def _warmup_model(self):
        """Прогрев модели для первого быстрого вызова"""
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model.predict(dummy_frame, verbose=False)
        print("Модель прогрета")

    def detect(self,frame):
        """
        ДЕТЕКЦИЯ ТС В КАДРЕ
        Args:
            frame: numpy array (BGR)
        Returns:
            list[Detecor]: детекции
        """
        results = self.model.predict(frame,device=self.device,conf=self.conf_thresh,verbose=False)
        detections = []
        # Проверяем, есть ли детекции
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # Получаем данные детекций
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            # Векторизируем фильтрацию по классам
            mask = np.isin(classes_ids, self.classes)
            boxes = boxes[mask]
            scores = scores[mask]
            classes_ids = classes_ids[mask]
            # Фильтруем по классам
            for box, score, class_id in zip(boxes, scores, classes_ids):
            # if class_id in self.classes:
                detection = Detection(
                    bbox_xyxy=box,
                    score=float(score),
                    class_id=int(class_id)
                )
                detections.append(detection)

        return detections