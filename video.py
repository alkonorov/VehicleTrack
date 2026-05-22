import cv2 as cv
import os

# VIDEO_PATH = '/home/alkon/code/test_tracker/data'
# VIDEO_NAME = 'test_video.mp4'


class VideoStream:
    def __init__(self,video_path):
        """
        Инициализация воспроизведения видео

        Args:
            video_path: путь к видео
        """
        self.video_path = video_path
        self.cap = None
        self.video_fps = None
        self.width = None
        self.height = None
        # self.frame = None
        self.frame_count = 0
        self.total_frames = None
        self.isOpened = False

    def open_video(self):
        """ Отрытие видео """
        self.cap = cv.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            print(f'Не удалось открыть видео: {self.video_path}')
            return False
        self.isOpened = True

        # Параметры видео
        self.video_fps = self.cap.get(cv.CAP_PROP_FPS)
        self.width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.frame_count = 0

        print(f"Видео открыто: {self.width}x{self.height}, Кадров: {self.total_frames}")
        return True

    def read(self):
        """
        Чтение кадра

        Returns:
            tuple(ret,frame)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
        return ret, frame



    def size(self):
        """
        Получение размера кадра

        Returns:
            tuple(width, height)
        """
        return self.width, self.height

    def release(self):
        """
        Высвобождение ресурсов
        """
        if self.cap:
            self.cap.release()
            self.isOpened = False
            print('Ресурсы видео освобождены')




