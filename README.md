🎯 Face Recognition Project

Dự án nhận diện người trong video stream theo thời gian thực, sử dụng HuggingFace Transformers và OpenCV.
Ứng dụng phù hợp cho các bài toán giám sát an ninh, theo dõi khu vực, hoặc tự động lưu lại sự kiện khi có người xuất hiện.

📖 Giới thiệu dự án

Hệ thống này được xây dựng để:

Kết nối tới camera IP (RTSP) hoặc webcam.

Phát hiện đối tượng người bằng mô hình facebook/detr-resnet-101.

Kết hợp với thuật toán motion detection để đảm bảo chỉ lưu ảnh khi có chuyển động thực sự.

Lưu ảnh vào thư mục captures/ để phục vụ theo dõi, báo cáo hoặc xử lý thêm.

🛠️ Ứng dụng thực tế

Giám sát an ninh tại văn phòng, kho bãi, nhà xưởng.

Tự động lưu bằng chứng khi phát hiện có người đi vào khu vực.

Cơ sở hạ tầng mở rộng để tích hợp vào dashboard giám sát hoặc hệ thống cảnh báo.

🚀 Yêu cầu hệ thống

Python 3.12.3

pip (Python package manager)

Git

OpenCV hỗ trợ RTSP

(Tùy chọn) GPU CUDA để tăng tốc mô hình

📥 Cài đặt
1. Clone project
git clone https://gitlab.eton.vn/anh.levanduc/face-recognition.git
cd face-recognition

2. Tạo môi trường ảo (khuyến nghị)
python -m venv venv


Kích hoạt môi trường ảo:

Windows:

venv\Scripts\activate


Linux / MacOS:

source venv/bin/activate

3. Cài đặt dependencies
pip install --upgrade pip
pip install -r requirements.txt

▶️ Chạy ứng dụng
python app.py