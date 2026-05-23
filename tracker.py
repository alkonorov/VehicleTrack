from typing import List, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from utils.geometry import iou_matrix
from lap import lapjv



class KalmanTracker:
    """ 1 фильтр Калмана - 1 трек



    """

    def __init__(self, bbox_xyxy: np.ndarray):
        """
        Args:
            bbox_xyxy: начальный bbox [x1, y1, x2, y2]
        """
        # Размерности
        self.dim_x = 8  # состояние
        self.dim_z = 4  # наблюдение

        # Матрица перехода состояния F (предполагаем постоянную скорость)
        # x_new = x + vx * dt,  где dt = 1 кадр
        self.F = np.eye(self.dim_x)
        self.F[0, 4] = 1.0  # cx += vx
        self.F[1, 5] = 1.0  # cy += vy
        self.F[2, 6] = 1.0  # w  += vw
        self.F[3, 7] = 1.0  # h  += vh

        # Матрица наблюдения H (мы измеряем только позицию и размер, не скорость)
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0, 0] = 1.0  # cx
        self.H[1, 1] = 1.0  # cy
        self.H[2, 2] = 1.0  # w
        self.H[3, 3] = 1.0  # h

        # Ковариация шума процесса Q (насколько модель «неточна»)
        self.Q = np.eye(self.dim_x)
        self.Q[4:, 4:] *= 0.1   # скорости меняются медленно
        self.Q[:4, :4] *= 1.0    # позиция может меняться быстрее

        # Ковариация шума измерений R (насколько «шумные» детекции)
        self.R = np.eye(self.dim_z) * 1.0

        # Ковариация ошибки оценки P
        self.P = np.eye(self.dim_x)
        self.P[4:, 4:] *= 100.0  # начальная неопределённость скоростей — высокая
        self.P[:4, :4] *= 10.0

        # Начальное состояние
        self.x = np.zeros(self.dim_x)
        z = self._bbox_to_z(bbox_xyxy)
        self.x[:4] = z  # позиция из bbox
        # скорости = 0 (пока не знаем)

    @staticmethod
    def _bbox_to_z(bbox_xyxy: np.ndarray) -> np.ndarray:
        """Конвертация [x1,y1,x2,y2] → [cx, cy, w, h]"""
        x1, y1, x2, y2 = bbox_xyxy
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return np.array([cx, cy, w, h])

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> np.ndarray:
        """Конвертация [cx, cy, w, h] → [x1, y1, x2, y2]"""
        cx, cy, w, h = z
        # Защита от отрицательных размеров
        w = max(w, 1.0)
        h = max(h, 1.0)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self) -> np.ndarray:
        """
        Шаг предсказания: сдвигаем состояние по модели движения.
        Returns:
            предсказанный bbox [x1, y1, x2, y2]
        """
        # x = F * x
        self.x = self.F @ self.x
        # P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self._z_to_bbox(self.x[:4])

    def update(self, bbox_xyxy: np.ndarray):
        """
        Шаг коррекции: уточняем состояние по реальной детекции.
        Args:
            bbox_xyxy: измеренный bbox [x1, y1, x2, y2]
        """
        z = self._bbox_to_z(bbox_xyxy)

        # Инновация (разница между измерением и предсказанием)
        y = z - self.H @ self.x

        # Ковариация инновации
        S = self.H @ self.P @ self.H.T + self.R

        # Коэффициент Калмана
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Обновление состояния
        self.x = self.x + K @ y

        # Обновление ковариации
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        """Текущий bbox [x1, y1, x2, y2] из состояния."""
        return self._z_to_bbox(self.x[:4])


@dataclass
class Track:
    track_id: int
    bbox_xyxy: np.ndarray  # [x1, y1, x2, y2]
    score: float
    class_id: int
    hits: int  # сколько кадров подряд был сматчен
    time_since_update: int  # сколько кадров без матча
    history: deque = field(default_factory=lambda: deque( maxlen=100))
    kalman: KalmanTracker = field(default=None, repr=False)


class ByteTracker:
    """Реализация ByteTrack"""
    def __init__(self,
                 hight_thresh=0.5,
                 low_thresh=0.1,
                 iou_thresh=0.3,
                 max_age=30, #кадры до удаления трека
                 min_hits=3 # мэтчи для подтверждения трека
                 ):
        self.hight_thresh = hight_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = [] # активные треки
        self.next_track_id = 1

    # @staticmethod
    def _get_center(self, bbox):
        """ Центр bbox"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return (cx, cy)

    def _match(self,tracks,detections, iou_threshold):
        """
        Мэтчинг треков и детекций (венгерский алгоритм)
        :return:
            matched: list[(track_idx, det_idx)]
            unmatched_tracks: list[(track_idx)]
            unmatched_dets: list[(det_idx)]
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_boxes = np.array([t.bbox_xyxy for t in tracks])
        det_boxes = np.array([d.bbox_xyxy for d in detections])
        iou_mat = iou_matrix(track_boxes, det_boxes)


        cost_matrix = 1.0 - iou_mat

        # lap.lapjv возвращает (cost, x, y)
        # x[i] = индекс столбца, назначенного строке i (-1 если не назначен)
        # y[j] = индекс строки, назначенной столбцу j (-1 если не назначен)

        cost, x, y = lapjv(cost_matrix,extend_cost=True,cost_limit=1.0 - iou_threshold)

        matched = []
        unmatched_tracks = []
        unmatched_dets = []

        for i in range(len(tracks)):
            if x[i] >= 0:
                matched.append((i, int(x[i])))
            else:
                unmatched_tracks.append(i)

        for j in range(len(detections)):
            if y[j] < 0:
                unmatched_dets.append(j)

        return matched, unmatched_tracks, unmatched_dets

    def _update_track(self, track: Track, det):
        """Обновить трек по детекции (коррекция Калмана + обновление полей)"""
        # Коррекция фильтра Калмана
        # track.kalman.update(det.bbox_xyxy)
        # Берём скорректированный bbox
        # track.bbox_xyxy = track.kalman.get_state()
        track.kalman.update(det.bbox_xyxy)


        track.score = det.score
        track.class_id = det.class_id
        track.hits += 1
        track.time_since_update = 0
        track.history.append(self._get_center(track.bbox_xyxy))


    def update(self, detections):
        """

        Args:
            detections: list[Detection] — все детекции с текущего кадра
                        (включая низкоуверенные! conf_thresh в детекторе
                         должен быть = low_thresh)

        Returns:
            list[Track] — подтверждённые треки
        """

        for track in self.tracks:
            predicted_bbox = track.kalman.predict()
            if track.time_since_update <= 3:
                track.bbox_xyxy = predicted_bbox
            if track.time_since_update > 0:
                track.kalman.x[4:8] *= 0.75  # демпфирование vx,vy,vw,vh

        high_dets = [d for d in detections if d.score >= self.hight_thresh]
        low_dets = [d for d in detections if self.hight_thresh > d.score >= self.low_thresh]

        # высокие детекции
        matched_1, unmatched_tracks_1, unmatched_high_dets = self._match(self.tracks, high_dets, self.iou_thresh)
        #
        # обновляем трек для смэтченых пар
        for track_idx, det_idx  in matched_1:
            self._update_track(self.tracks[track_idx], high_dets[det_idx])
            # track = self.tracks[track_idx]
            # det = high_dets[det_idx]
            #
            # track.bbox_xyxy = det.bbox_xyxy
            # track.score = det.score
            # track.class_id = det.class_id
            # track.hits += 1
            # track.time_since_update = 0
            # track.history.append(self._get_center(track.bbox_xyxy))


        # низкие детекции
        remaining_tracks = [  self.tracks[i] for i in unmatched_tracks_1 ]

        matched_2, unmatched_tracks_2, _ = self._match(remaining_tracks, low_dets, self.iou_thresh)

        for track_idx, det_idx  in matched_2:
            self._update_track(remaining_tracks[track_idx], low_dets[det_idx])
            # track = remaining_tracks[track_idx]
            # det = low_dets[det_idx]
            #
            # track.bbox_xyxy = det.bbox_xyxy
            # track.score = det.score
            # track.class_id = det.class_id
            # track.hits += 1
            # track.time_since_update = 0
            # track.history.append(self._get_center(track.bbox_xyxy))

        # несмэтченые
        for track_idx in unmatched_tracks_2:
            # remaining_tracks[track_idx].time_since_update += 1
            track = remaining_tracks[track_idx]
            track.time_since_update += 1

        # удаление мертвых треков
        self.tracks = [t for t in self.tracks   if t.time_since_update <= self.max_age]

        # ===== ШАГ 5: Создать новые треки из несматченных ВЫСОКИХ детекций =====
        # (только из высоких! низкие — ненадёжные, из них треки не создаём)
        # Для каждой несматченной высокой детекции:
        for det_idx in unmatched_high_dets:
            det = high_dets[det_idx]
            new_track = Track(
              track_id=self.next_track_id,
              bbox_xyxy=det.bbox_xyxy,
              score=det.score,
              class_id=det.class_id,
              hits=1,
              time_since_update=0,
              history=deque([self._get_center(det.bbox_xyxy)], maxlen=100),
              kalman=KalmanTracker(det.bbox_xyxy),
            )
            self.tracks.append(new_track)
            self.next_track_id += 1

        #
        return [t for t in self.tracks if t.hits >= self.min_hits]

    def get_all_track_ids(self) -> set:
        """Возвращает множество  живых треков (для очистки в counter)."""
        return {t.track_id for t in self.tracks}
