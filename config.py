import os

# ==========================================
# Camera Config
# ==========================================
USE_IP_CAMERA = True
RTSP_URLS = [
    "rtsp://admin:2025@EtonCam@192.168.60.1:554/Streaming/Channels/2802",
    "rtsp://admin:2025@EtonCam@192.168.60.1:554/Streaming/Channels/602",
    "rtsp://admin:2025@EtonCam@192.168.60.1:554/Streaming/Channels/2402",
    "rtsp://admin:2025@EtonCam@192.168.60.1:554/Streaming/Channels/2902",
    "rtsp://admin:2025@EtonCam@192.168.60.2:554/Streaming/Channels/102",
    "rtsp://admin:2025@EtonCam@192.168.60.2:554/Streaming/Channels/1102",
    "rtsp://admin:2025@EtonCam@192.168.60.2:554/Streaming/Channels/1802",
    "rtsp://admin:2025@EtonCam@192.168.60.2:554/Streaming/Channels/2402"
]

# ==========================================
# Capture Config
# ==========================================
CAPTURE_DIR = "captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)
CAPTURE_COOLDOWN = 5  # giây, thời gian chờ giữa 2 lần chụp

# ==========================================
# Detection Config
# ==========================================
BATCH_SIZE = 8  # Tăng lên bằng số lượng camera
MAX_QUEUE_SIZE = 20 # Tăng kích thước hàng đợi để có thêm bộ đệm
PROCESS_EVERY_N_FRAMES = 5 # Chỉ xử lý AI mỗi 5 frame, giảm tải 80%
SCALE_FACTOR = 1.0
CONFIDENCE_THRESHOLD = 0.9
MOTION_THRESHOLD = 10 # Giảm ngưỡng để nhạy hơn với chuyển động nhỏ

# ==========================================
# Logging Config
# ==========================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")
