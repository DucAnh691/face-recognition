import os
import cv2
import time
import logging
from datetime import datetime
from threading import Lock
from config import CAPTURE_DIR, CAPTURE_COOLDOWN

class CaptureSaver:
    def __init__(self):
        # Lưu thời điểm chụp cuối cùng cho mỗi camera
        self.last_capture_times = {}
        self._lock = Lock()

    def save(self, frame, cam_id):
        """Lưu ảnh nếu đã đủ thời gian cooldown cho camera này."""
        with self._lock:
            current_time = time.time()
            last_time = self.last_capture_times.get(cam_id, 0)

            # Nếu chưa đủ thời gian chờ, không làm gì cả
            if current_time - last_time < CAPTURE_COOLDOWN:
                return None
            
            # Nếu đủ thời gian, cập nhật thời điểm và lưu ảnh
            self.last_capture_times[cam_id] = current_time
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            # Tên tệp không cần track_id nữa
            capture_path = os.path.join(CAPTURE_DIR, f"cam{cam_id}_{timestamp}.jpg")
            cv2.imwrite(capture_path, frame)
            logging.info(f"Ảnh lưu tại: {capture_path}")
            return capture_path
