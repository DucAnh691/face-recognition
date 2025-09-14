# --- QUAN TRỌNG: Đặt biến môi trường để sử dụng TCP cho RTSP ---
# Phải đặt trước khi import cv2 để có hiệu lực
import os
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

# Import necessary libraries
import cv2
import torch
import time
from datetime import datetime
import threading
from transformers import pipeline
from PIL import Image

# --- Class để đọc video trong luồng riêng, giảm độ trễ ---
class VideoStreamWidget:
    def __init__(self, src=0):
        # Sử dụng cv2.CAP_FFMPEG để hỗ trợ tốt hơn cho các luồng mạng
        self.capture = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            print(f"Lỗi: Không thể mở nguồn video: {src}")
            self.status = False
            return

        # Bắt đầu luồng để đọc các khung hình từ video stream
        self.status, self.frame = self.capture.read()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Đọc khung hình tiếp theo trong một luồng riêng
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(.01) # Giảm tải CPU

    def read(self):
        # Trả về khung hình gần nhất
        return self.status, self.frame


# Load the face detection pipeline
# Kiểm tra xem có GPU (CUDA) không và sử dụng nếu có
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Sử dụng model yolos-tiny nhẹ hơn và chuyên cho phát hiện đối tượng
# Sử dụng model lớn hơn để tăng độ chính xác, không ưu tiên tốc độ
# face_detector = pipeline("object-detection", model="hustvl/yolos-tiny", device=device)
face_detector = pipeline("object-detection", model="facebook/detr-resnet-101", device=device)


def recognize_faces_realtime():
    """
    Recognizes faces from a webcam in real-time and displays the output.
    """
    # --- Cấu hình Camera ---
    # Để sử dụng webcam, đặt USE_IP_CAMERA = False
    # Để sử dụng camera IP, đặt USE_IP_CAMERA = True và điền thông tin rtsp_url
    USE_IP_CAMERA = True

    # URL RTSP cho camera IP (ví dụ cho Hikvision)
    # !!! QUAN TRỌNG: Thay thế <user> và <password> bằng thông tin đăng nhập chính xác của bạn.
    # rtsp://<user>:<password>@<ip_address>:<port>/Streaming/Channels/<channel_id>
    # Thử dùng luồng phụ (sub-stream) '2402' để nhẹ hơn và ổn định hơn
    # Ví dụ của bạn:
    # rtsp_url = "rtsp://admin:YourCorrectPassword@192.168.60.1:554/Streaming/Channels/2701"
    rtsp_url = "rtsp://admin:2025@EtonCam@192.168.60.2:554/Streaming/Channels/1102"  # <-- SỬA DÒNG NÀY

    if USE_IP_CAMERA:
        video_stream = VideoStreamWidget(rtsp_url)
    else:
        # Mở webcam mặc định (chỉ số 0)
        video_stream = VideoStreamWidget(0)

    if not video_stream.capture.isOpened():
        print(f"Lỗi: Không thể mở nguồn video. Vui lòng kiểm tra lại cấu hình camera.")
        return
    
    # --- Cấu hình cho việc chụp ảnh ---
    CAPTURE_DIR = "captures"
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    # Thời gian chờ (giây) giữa các lần chụp ảnh để tránh lưu quá nhiều file
    CAPTURE_COOLDOWN = 5  # 5 giây
    last_capture_time = 0

    print("Đang mở webcam... Nhấn 'q' trên cửa sổ video để thoát.")

    frame_count = 0
    # Chỉ xử lý mỗi 3 khung hình để tăng hiệu suất
    PROCESS_EVERY_N_FRAMES = 1  # Xử lý mọi khung hình để có độ chính xác cao nhất
    # Giảm kích thước khung hình để xử lý nhanh hơn
    SCALE_FACTOR = 1.0  # Sử dụng kích thước gốc để không bỏ lỡ chi tiết
    
    # Khởi tạo detections để tránh lỗi UnboundLocalError
    detections = []

    # --- Cấu hình bộ lọc để tăng độ chính xác ---
    CONFIDENCE_THRESHOLD = 0.9  # Ngưỡng tin cậy cao cho model chính xác hơn
    # MIN_AREA_PERCENT đã được loại bỏ để không lọc mất các đối tượng nhỏ

    # --- Cấu hình phát hiện chuyển động ---
    previous_frame_gray = None
    MOTION_THRESHOLD = 30  # Ngưỡng để xác định có chuyển động trong bounding box

    while True:
        # Đọc từng khung hình từ webcam
        ret, frame = video_stream.read()
        if not ret:
            print("Lỗi: Không thể nhận khung hình.")
            break

        # Tạo một bản sao của khung hình để vẽ lên, tránh ảnh hưởng đến khung hình gốc
        # được sử dụng trong các lần lặp tiếp theo (khi bỏ qua frame)
        display_frame = frame.copy()

        # Lấy kích thước khung hình để tính toán diện tích tối thiểu
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_frame_gray = cv2.GaussianBlur(current_frame_gray, (21, 21), 0)

        motion_detected_in_frame = False

        frame_count += 1
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            # Giảm kích thước khung hình
            small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

            # Chuyển đổi sang PIL Image để đưa vào model
            pil_image = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
            
            # Phát hiện đối tượng
            detections = face_detector(pil_image)

        # So sánh với khung hình trước để phát hiện chuyển động
        if previous_frame_gray is not None:
            frame_delta = cv2.absdiff(previous_frame_gray, current_frame_gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_detected_in_frame = thresh.sum() > 0

        # Vẽ các hộp bao quanh đối tượng được phát hiện (vẽ trên mọi frame để video mượt hơn)
        for detection in detections:
            box = detection["box"]
            label = detection["label"]
            score = detection["score"]

            # Áp dụng các bộ lọc để tăng độ chính xác
            if label == "person" and score > CONFIDENCE_THRESHOLD: # Lọc theo độ tin cậy
                # Quy đổi tọa độ về kích thước khung hình gốc
                (xmin, ymin, xmax, ymax) = (int(box["xmin"] / SCALE_FACTOR), int(box["ymin"] / SCALE_FACTOR),
                                            int(box["xmax"] / SCALE_FACTOR), int(box["ymax"] / SCALE_FACTOR))
                
                # Chỉ xử lý nếu có chuyển động trong khung hình
                if motion_detected_in_frame:
                    # Kiểm tra xem có chuyển động bên trong bounding box không
                    roi_motion = thresh[ymin:ymax, xmin:xmax]
                    motion_in_box = cv2.countNonZero(roi_motion) > MOTION_THRESHOLD if roi_motion.size > 0 else False

                    if motion_in_box:
                        cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Person: {score:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Tác vụ chụp ảnh (chỉ khi tất cả điều kiện đều đúng)
                        current_time = time.time()
                        if current_time - last_capture_time > CAPTURE_COOLDOWN:
                            last_capture_time = current_time
                            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
                            capture_path = os.path.join(CAPTURE_DIR, f"capture_{timestamp_str}.jpg")
                            cv2.imwrite(capture_path, display_frame)
                            print(f"Đã phát hiện người và lưu ảnh tại: {capture_path}")
        
        # Hiển thị khung hình đã xử lý
        cv2.imshow('Real-time Face Detection', display_frame)
        
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Cập nhật khung hình trước đó
        previous_frame_gray = current_frame_gray
    
    # Giải phóng webcam và đóng tất cả cửa sổ
    video_stream.capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces_realtime()
