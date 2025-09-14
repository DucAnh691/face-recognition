import os
import cv2
import time
import logging
from datetime import datetime
from threading import Lock
from config import CAPTURE_DIR, CAPTURE_COOLDOWN

class CaptureSaver:
    def __init__(self):
        self.last_capture_time = 0
        self._lock = Lock()

    def save(self, frame):
        """Lưu frame thành ảnh khi phát hiện đối tượng."""
        current_time = time.time()
        if current_time - self.last_capture_time < CAPTURE_COOLDOWN:
            return None

        with self._lock:
            # Kiểm tra lại điều kiện sau khi có được lock
            if current_time - self.last_capture_time < CAPTURE_COOLDOWN:
                return None
            self.last_capture_time = current_time
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            capture_path = os.path.join(CAPTURE_DIR, f"capture_{timestamp}.jpg")
            cv2.imwrite(capture_path, frame)
            logging.info(f"Ảnh lưu tại: {capture_path}")
            return capture_path
