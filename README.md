# 🎯 Hệ thống Giám sát An ninh Thông minh với YOLOv8 và ByteTrack

Dự án xây dựng một hệ thống giám sát an ninh hoàn chỉnh, có khả năng phát hiện và theo dõi đối tượng "người" trong thời gian thực từ nhiều nguồn camera khác nhau.

---

## 📖 Tính năng chính

- **Phát hiện và Theo dõi Người Chính xác**: Sử dụng model **YOLOv8s** mạnh mẽ để phát hiện và thuật toán **ByteTrack** để theo dõi, gán ID duy nhất cho mỗi người xuất hiện trong khung hình.
- **Hỗ trợ Đa luồng Camera**: Có khả năng kết nối và xử lý đồng thời nhiều luồng video từ camera IP (RTSP).
- **Chụp ảnh Tự động**: Khi phát hiện có người, hệ thống sẽ tự động lưu lại một ảnh vào thư mục `captures/`.
- **Kiến trúc Đa luồng Bền bỉ**: Mỗi camera được xử lý trong một luồng riêng biệt. Đặc biệt, mỗi luồng sẽ tải một bản sao của model YOLOv8 để đảm bảo trạng thái theo dõi (tracking state) của các camera không xung đột với nhau, giúp hệ thống hoạt động ổn định.
- **Tự động Phục hồi**: Một luồng giám sát chuyên dụng sẽ tự động phát hiện và khởi động lại các luồng camera bị lỗi, đảm bảo hệ thống hoạt động 24/7.

## 💡 Công nghệ sử dụng

- **Python 3.10+**: Ngôn ngữ lập trình chính của dự án.(Khuyên dùng 3.12.3)
- **OpenCV**: Thư viện mã nguồn mở hàng đầu cho các tác vụ thị giác máy tính, được sử dụng để đọc và xử lý luồng video.
- **Ultralytics YOLOv8**: Framework và model AI cốt lõi cho việc phát hiện đối tượng. Dự án sử dụng phiên bản `yolov8s.pt` (small) để cân bằng giữa tốc độ và độ chính xác. ( Có thể thử với YOLOv10 nhưng v8 sẽ an toàn hơn)
  - Tài liệu chính thức của YOLOv8 : https://docs.ultralytics.com/?utm_source=chatgpt.com#yolo-a-brief-history
- **ByteTrack**: Thuật toán theo dõi đối tượng hiệu suất cao, được tích hợp sẵn trong YOLOv8 để gán và duy trì ID cho các đối tượng qua các khung hình.
  - Bài báo khoa học về ByteTrack : https://github.com/FoundationVision/ByteTrack

## 📂 Cấu trúc thư mục

```
face-recognition/
├── captures/             # Thư mục chứa các ảnh chụp được
├── logs/                 # Thư mục chứa file log hoạt động
├── core/                 # Chứa các module xử lý cốt lõi
│   ├── __init__.py
│   ├── camera.py         # Lớp đọc video stream trong luồng riêng để giảm độ trễ
│   ├── detector.py       # Hàm tải model YOLOv8
│   ├── saver.py          # Lớp xử lý logic lưu ảnh và cơ chế cooldown
│   ├── service.py        # Lớp dịch vụ chính, điều phối các luồng camera và giám sát
│   └── utils.py          # Các hàm tiện ích (ví dụ: thiết lập logging)
├── app.py                # Điểm khởi chạy chính của ứng dụng
├── config.py             # File cấu hình tập trung (URL camera, ngưỡng, v.v.)
└── README.md             # Tài liệu hướng dẫn dự án (chính là file này)
```

## 📥 Hướng dẫn cài đặt chi tiết

### Bước 0: Chuẩn bị các công cụ cần thiết

Trước khi bắt đầu, hãy đảm bảo bạn đã cài đặt **Git** và **Python**.

### Bước 1: Clone repo

- git clone https://gitlab.eton.vn/anh.levanduc/face-recognition.git
- cd face-recognition


### Bước 2: Tạo và kích hoạt môi trường ảo


1.  **Tạo môi trường ảo** (đặt tên là `venv`):
    
    python -m venv venv
    
2.  **Kích hoạt môi trường ảo**:
    -   Trên **Windows** (dùng Command Prompt hoặc PowerShell):
        
        venv\Scripts\activate
        
    -   Trên **Linux** hoặc **macOS**:
        
        source venv/bin/activate
        

### Bước 3: Cài đặt các thư viện

pip install -r requirements.txt


## ⚙️ Cấu hình hệ thống


-   **`RTSP_URLS`**: Đây là danh sách các địa chỉ RTSP của camera IP. 
-   **`USE_IP_CAMERA`**:
    -   Đặt là `True` để hệ thống sử dụng danh sách `RTSP_URLS`.
    -   Đặt là `False` để sử dụng webcam mặc định của máy tính (hữu ích cho việc kiểm thử).
-   **`CAPTURE_COOLDOWN`**: Thời gian (tính bằng giây) mà hệ thống sẽ chờ trước khi chụp một ảnh mới cho cùng một camera.
-   **`PROCESS_EVERY_N_FRAMES`**: Để giảm tải cho CPU, hệ thống sẽ chỉ chạy model AI trên mỗi N khung hình. Tăng giá trị này (ví dụ: 10, 15) nếu bạn thấy hệ thống bị giật, lag.
-   **`CONFIDENCE_THRESHOLD`**: Ngưỡng tin cậy (từ 0.0 đến 1.0). Chỉ những đối tượng "người" có độ tin cậy lớn hơn giá trị này mới được xử lý và hiển thị.

## ▶️ Khởi chạy

python app.py


**Khi chương trình chạy:**
-   Các cửa sổ tương ứng với mỗi camera sẽ hiện lên, hiển thị video stream và các đối tượng được phát hiện.
-   Khi có người được phát hiện, ảnh chụp sẽ được lưu vào thư mục `captures/`.
-   Mọi hoạt động, cảnh báo và lỗi sẽ được ghi vào tệp `logs/app.log`.
-   Để dừng chương trình, bạn có thể nhấn phím `q` trên bất kỳ cửa sổ camera nào.

---
