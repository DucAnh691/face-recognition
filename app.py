import os
# QUAN TRỌNG: Đặt biến môi trường để dùng TCP cho RTSP, giảm lỗi mất gói tin
# Phải đặt trước khi import cv2 (thông qua các module khác)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

from core.service import FaceRecognitionService
import logging

if __name__ == "__main__":
    service = FaceRecognitionService()
    try:
        service.run()
    finally:
        service.shutdown()
