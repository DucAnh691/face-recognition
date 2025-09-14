import cv2
import threading
import time
import logging

class VideoStreamWidget:
    """Đọc video từ camera trong một thread riêng để giảm độ trễ."""
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            logging.error(f"Không thể mở video source: {src}")
            self.status, self.frame = False, None
            return

        self.status, self.frame = self.capture.read()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(0.01)  # giảm tải CPU

    def read(self):
        return self.status, self.frame

    def release(self):
        self.capture.release()
