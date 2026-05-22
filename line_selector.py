import cv2 as cv


class LineSelector:
    """Интерактивный выбор линии подсчёта на первом кадре"""

    def __init__(self):
        self.points = []
        self.done = False

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(self.points) < 2:
            self.points.append((x, y))
            print(f"  Точка {len(self.points)}: ({x}, {y})")
            if len(self.points) == 2:
                self.done = True

    def select(self, frame) -> tuple:
        """
        Показать кадр и дать пользователю выбрать 2 точки линии.

        Returns:
            (line_start, line_end) или None если отменено
        """
        print("\n" + "=" * 50)
        print("🖱  Кликните 2 точки для линии подсчёта")
        print("    [ESC] — отмена, [R] — сброс")
        print("=" * 50)

        window_name = "Select Counting Line"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.setMouseCallback(window_name, self._mouse_callback)

        display = frame.copy()

        while not self.done:
            show = display.copy()

            # Рисуем уже поставленные точки
            for i, pt in enumerate(self.points):
                cv.circle(show, pt, 6, (0, 0, 255), -1)
                cv.putText(show, f"P{i + 1}", (pt[0] + 10, pt[1] - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Если 2 точки — рисуем линию
            if len(self.points) == 2:
                cv.line(show, self.points[0], self.points[1], (0, 0, 255), 3)

            cv.putText(show, "Click 2 points for counting line | ESC=cancel | R=reset",
                       (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv.imshow(window_name, show)
            key = cv.waitKey(30) & 0xFF

            if key == 27:  # ESC
                cv.destroyWindow(window_name)
                return None
            elif key == ord('r'):
                self.points = []
                self.done = False

        cv.destroyWindow(window_name)
        print(f"  ✅ Линия: {self.points[0]} -> {self.points[1]}")
        return self.points[0], self.points[1]