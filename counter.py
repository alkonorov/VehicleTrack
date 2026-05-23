import numpy as np
from collections import defaultdict


class VehicleCounter:
    """
    Подсчёт транспорта по ID

    Стратегия подсчета трека:
    1. Трек должен просуществовать min_age кадров
    2. Трек должен пройти min_displacement пикселей (исключаем дрожание на месте)
    3. Трек должен иметь устойчивое направление (не менее direction_consistency% кадров
       движения в одну сторону — фильтруем "метания")

    считаем трек если все выполняется
    """

    def __init__(
            self,
            direction_axis='diagonal',
            min_age=15,  # минимум кадров существования трека
            min_displacement=50,  # минимум пикселей смещения от start до end
            direction_consistency=0.7,  # доля кадров с движением в «победившую» сторону
            min_movement_frames=5,  # сколько кадров с реальным движением нужно
    ):
        """
        Args:
            direction_axis: 'x', 'y' или 'diagonal'
            min_age: сколько кадров трек должен прожить до засчитывания
            min_displacement: минимальный пробег в пикселях (фильтр стоячих машин)
            direction_consistency: порог согласованности направления (0.5–1.0)
            min_movement_frames: минимум кадров, где было зафиксировано движение
        """
        self.direction_axis = direction_axis
        self.min_age = min_age
        self.min_displacement = min_displacement
        self.direction_consistency = direction_consistency
        self.min_movement_frames = min_movement_frames

        # Множество уже посчитанных ID
        self.counted_ids = set()

        # Для анализа направления:
        self.track_direction_votes = defaultdict(list)

        self.count_a = 0
        self.count_b = 0

        self.just_counted = set()


    # def _get_direction(self, history) -> str:
    #     """
    #     Определить направление движения по истории центров.
    #     """
    #     if len(history) < 2:
    #         return 'A'
    #
    #     first = np.array(history[0])
    #     last = np.array(history[-1])
    #     delta = last - first
    #
    #     if self.direction_axis == 'x':
    #         return 'A' if delta[0] > 0 else 'B'
    #     elif self.direction_axis == 'y':
    #         return 'A' if delta[1] > 0 else 'B'
    #     elif self.direction_axis == 'diagonal':
    #         proj = delta[0] - delta[1]
    #         return 'A' if proj > 0 else 'B'
    #
    #     return 'A'

    def _get_frame_direction(self, p1, p2) -> str:
        """
        Направление движения между двумя соседними точками.
        """
        delta = np.array(p2) - np.array(p1)
        if self.direction_axis == 'x':
            return 'A' if delta[0] > 0 else 'B'
        elif self.direction_axis == 'y':
            return 'A' if delta[1] > 0 else 'B'
        elif self.direction_axis == 'diagonal':
            proj = delta[0] - delta[1]
            return 'A' if proj > 0 else 'B'
        return 'A'

    def _compute_displacement(self, history) -> float:
        """
        Вычислить чистое смещение от первой до последней точки (евклидово расстояние)
        """
        if len(history) < 2:
            return 0.0
        start = np.array(history[0])
        end = np.array(history[-1])
        return float(np.linalg.norm(end - start))

    def _is_direction_consistent(self, direction_votes: list) -> bool:
        """
        Провереяем устойчивость направления.
        direction_votes: список 'A'/'B' для каждого кадра с движением
        """
        if len(direction_votes) < self.min_movement_frames:
            return False

        count_a = direction_votes.count('A')
        count_b = direction_votes.count('B')
        total = count_a + count_b

        if total == 0:
            return False

        majority_ratio = max(count_a, count_b) / total
        return majority_ratio >= self.direction_consistency

    def _majority_direction(self, direction_votes: list) -> str:
        """Направление"""
        count_a = direction_votes.count('A')
        count_b = direction_votes.count('B')
        return 'A' if count_a >= count_b else 'B'

    def _is_track_mature(self, track) -> bool:
        """
        Проверяем все критерии
        True- засчитываем трек
        """
        tid = track.track_id
        history = list(track.history)

        # минимальный возраст
        if len(history) < self.min_age:
            return False

        # минимальное смещение
        displacement = self._compute_displacement(history)
        if displacement < self.min_displacement:
            return False

        # устойчивость направления
        votes = self.track_direction_votes.get(tid, [])
        if not self._is_direction_consistent(votes):
            return False

        return True

    def update(self, tracks: list,active_track_ids:set = None) -> set:
        """
        Обновить счётчик и накопить статистику по направлениям

        Args:
            tracks: список Track объектов (все треки, включая неподтверждённые)
            active_track_ids - id всех живых треков(для очистки )

        Returns:
            set: ID треков, посчитанных на этом кадре
        """
        self.just_counted = set()

        for track in tracks:
            tid = track.track_id

            # Уже считали — пропускаем
            if tid in self.counted_ids:
                continue

            history = list(track.history)
            if len(history) < 2:
                continue

            # Накапливаем «голоса» за направление на каждом кадре
            # (сравниваем две последние точки, если было реальное движение)
            if len(history) >= 2:
                p1 = history[-2]
                p2 = history[-1]
                disp = np.linalg.norm(np.array(p2) - np.array(p1))
                # Учитываем только кадры, где было движение > 1 пикселя
                if disp > 1.0:
                    frame_dir = self._get_frame_direction(p1, p2)
                    self.track_direction_votes[tid].append(frame_dir)

            # Проверяем зрелость трека
            if not self._is_track_mature(track):
                continue

            # Определяем итоговое направление по накопленным голосам
            direction = self._majority_direction(self.track_direction_votes[tid])

            # Засчитываем трек
            if direction == 'A':
                self.count_a += 1
            else:
                self.count_b += 1

            self.counted_ids.add(tid)
            self.just_counted.add(tid)

            # Очищаем память для посчитаного трека
            if tid in self.track_direction_votes:
                del self.track_direction_votes[tid]

            total = self.count_a + self.count_b
            print(f"  ID:{tid} → {direction} | "
                  f"Total: {total} (A:{self.count_a} B:{self.count_b})")

            if active_track_ids is not None:
                dead_ids = [
                    tid for tid in self.track_direction_votes
                    if tid not in active_track_ids
                ]
                for tid in dead_ids:
                    del self.track_direction_votes[tid]


        return self.just_counted

    def get_stats(self) -> dict:
        return {
            "total": self.count_a + self.count_b,
            "direction_a": self.count_a,
            "direction_b": self.count_b,
        }

    def get_display_text(self) -> list:
        total = self.count_a + self.count_b
        return [
            f"TOTAL: {total}",
            f"From us: {self.count_a}  To us : {self.count_b}"
        ]

    def reset(self):
        """Сброс всех состояний"""
        self.counted_ids.clear()
        self.track_direction_votes.clear()
        self.count_a = 0
        self.count_b = 0
        self.just_counted.clear()