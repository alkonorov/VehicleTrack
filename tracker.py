from typing import List, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from utils.geometry import iou_matrix
from lap import lapjv


@dataclass
class Track:
    track_id: int
    bbox_xyxy: np.ndarray  # [x1, y1, x2, y2]
    score: float
    class_id: int
    hits: int  # сколько кадров подряд был сматчен
    time_since_update: int  # сколько кадров без матча
    history: deque = field(default_factory=lambda: deque( maxlen=100))


class ByteTracker:
    """Реализация ByteTrack"""
    def __init__(self, hight_thresh=0.5,
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


    def update(self, detections):
        """

        Args:
            detections: list[Detection] — все детекции с текущего кадра
                        (включая низкоуверенные! conf_thresh в детекторе
                         должен быть = low_thresh)

        Returns:
            list[Track] — подтверждённые треки
        """

        high_dets = [d for d in detections if d.score >= self.hight_thresh]
        low_dets = [d for d in detections if self.hight_thresh > d.score >= self.low_thresh]

        # высокие детекции
        matched_1, unmatched_tracks_1, unmatched_high_dets = self._match(self.tracks, high_dets, self.iou_thresh)
        #
        # обновляем трек для смэтченых пар
        for track_idx, det_idx  in matched_1:
            track = self.tracks[track_idx]
            det = high_dets[det_idx]

            track.bbox_xyxy = det.bbox_xyxy
            track.score = det.score
            track.class_id = det.class_id
            track.hits += 1
            track.time_since_update = 0
            track.history.append(self._get_center(track.bbox_xyxy))

        # низкие детекции
        remaining_tracks = [  self.tracks[i] for i in unmatched_tracks_1 ]

        matched_2, unmatched_tracks_2, _ = self._match(remaining_tracks, low_dets, self.iou_thresh)

        for track_idx, det_idx  in matched_2:
            track = remaining_tracks[track_idx]
            det = low_dets[det_idx]

            track.bbox_xyxy = det.bbox_xyxy
            track.score = det.score
            track.class_id = det.class_id
            track.hits += 1
            track.time_since_update = 0
            track.history.append(self._get_center(track.bbox_xyxy))

        # несмэтченые
        for track_idx in unmatched_tracks_2:
            remaining_tracks[track_idx].time_since_update += 1



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
              history=deque([self._get_center(det.bbox_xyxy)], maxlen=100)
            )
            self.tracks.append(new_track)
            self.next_track_id += 1

        #
        return [t for t in self.tracks if t.hits >= self.min_hits]