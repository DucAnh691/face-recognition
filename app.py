import os
import cv2
import torch
import time
import threading
from datetime import datetime
from transformers import pipeline
from PIL import Image

# ==========================================
# Cấu hình chung
# ==========================================
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

USE_IP_CAMERA = True
RTSP_URL = "rtsp://admin:2025@EtonCam@192.168.60.1:554/Streaming/Channels/2802"

CAPTURE_DIR = "captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

CAPTURE_COOLDOWN = 5          # giây, thời gian chờ giữa 2 lần chụp
PROCESS_EVERY_N_FRAMES = 1    # xử lý mỗi N frame
SCALE_FACTOR = 1.0            # resize frame trước khi detect
CONFIDENCE_THRESHOLD = 0.9    # ngưỡng tin cậy tối thiểu
MOTION_THRESHOLD = 30         # ngưỡng điểm ảnh để xác định có chuyển động

# ==========================================
# Class đọc video stream trong thread riêng
# ==========================================
class VideoStreamWidget:
    """Đọc video từ camera trong một thread riêng để giảm độ trễ."""
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            print(f"[ERROR] Không thể mở video source: {src}")
            self.status = False
            return

        self.status, self.frame = self.capture.read()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        """Liên tục đọc frame từ camera."""
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(0.01)  # giảm tải CPU

    def read(self):
        """Trả về frame mới nhất."""
        return self.status, self.frame


# ==========================================
# Load model phát hiện đối tượng
# ==========================================
if torch.cuda.is_available():
    device = 0
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[INFO] Phát hiện GPU: {gpu_name}, sẽ chạy trên GPU.")
else:
    device = -1
    print("[INFO] Không có GPU, sẽ chạy trên CPU.")

face_detector = pipeline(
    "object-detection",
    model="facebook/detr-resnet-101",
    device=device
)


# ==========================================
# Hàm nhận diện real-time
# ==========================================
def recognize_faces_realtime():
    """
    Nhận diện người trong video stream (webcam hoặc camera IP).
    Khi phát hiện có người + chuyển động, lưu ảnh vào thư mục captures/.
    """
    src = RTSP_URL if USE_IP_CAMERA else 0
    video_stream = VideoStreamWidget(src)

    if not video_stream.capture.isOpened():
        print("[ERROR] Không thể mở camera. Kiểm tra lại cấu hình.")
        return

    print("[INFO] Đang chạy... Nhấn 'q' để thoát.")

    last_capture_time = 0
    frame_count = 0
    previous_frame_gray = None
    detections = []

    while True:
        ret, frame = video_stream.read()
        if not ret:
            print("[ERROR] Không thể đọc frame.")
            break

        display_frame = frame.copy()
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_frame_gray = cv2.GaussianBlur(current_frame_gray, (21, 21), 0)

        motion_detected_in_frame = False
        frame_count += 1

        # Detect mỗi N frame
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            pil_image = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
            detections = face_detector(pil_image)

        # Phát hiện chuyển động
        if previous_frame_gray is not None:
            frame_delta = cv2.absdiff(previous_frame_gray, current_frame_gray)
            _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
            motion_detected_in_frame = thresh.sum() > 0

        # Vẽ bounding box
        for detection in detections:
            box, label, score = detection["box"], detection["label"], detection["score"]

            if label == "person" and score > CONFIDENCE_THRESHOLD:
                xmin, ymin, xmax, ymax = map(int, (
                    box["xmin"] / SCALE_FACTOR,
                    box["ymin"] / SCALE_FACTOR,
                    box["xmax"] / SCALE_FACTOR,
                    box["ymax"] / SCALE_FACTOR
                ))

                if motion_detected_in_frame:
                    roi_motion = thresh[ymin:ymax, xmin:xmax]
                    motion_in_box = cv2.countNonZero(roi_motion) > MOTION_THRESHOLD if roi_motion.size > 0 else False

                    if motion_in_box:
                        cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Person: {score:.2f}", (xmin, ymin - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Lưu ảnh
                        current_time = time.time()
                        if current_time - last_capture_time > CAPTURE_COOLDOWN:
                            last_capture_time = current_time
                            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
                            capture_path = os.path.join(CAPTURE_DIR, f"capture_{timestamp}.jpg")
                            cv2.imwrite(capture_path, display_frame)
                            print(f"[INFO] Ảnh lưu tại: {capture_path}")

        cv2.imshow('Real-time Detection', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        previous_frame_gray = current_frame_gray

    video_stream.capture.release()
    cv2.destroyAllWindows()


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    recognize_faces_realtime()
