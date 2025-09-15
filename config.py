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
CAPTURE_COOLDOWN = 10  # giây, thời gian chờ giữa 2 lần chụp

# ==========================================
# Detection Config
# ==========================================
# Xử lý AI mỗi N khung hình để giảm tải CPU.
PROCESS_EVERY_N_FRAMES = 10 # Tăng lên để giảm tải CPU hơn nữa
CONFIDENCE_THRESHOLD = 0.7 # Chỉ xử lý và vẽ đối tượng có độ tin cậy >= 70%
# MOTION_THRESHOLD không còn cần thiết

# ==========================================
# Logging Config
# ==========================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")
