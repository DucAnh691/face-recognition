import cv2
import logging
import time
import threading

from config import (
    RTSP_URLS, USE_IP_CAMERA, CONFIDENCE_THRESHOLD, PROCESS_EVERY_N_FRAMES, LOG_FILE
)
from core.camera import VideoStreamWidget
from core.detector import load_detector
from core.saver import CaptureSaver
from core.utils import setup_logging

class FaceRecognitionService:
    def __init__(self):
        setup_logging(LOG_FILE)
        # Không tải model ở đây nữa để tránh dùng chung giữa các luồng
        # self.detector = load_detector()
        self.saver = CaptureSaver()
        self.camera_threads = {}
        self.stop_event = threading.Event()
        
    def _camera_worker(self, cam_id, camera_src, window_name):
        """Luồng xử lý cho mỗi camera: đọc, theo dõi, phát hiện chuyển động và hiển thị."""
        # Mỗi luồng sẽ tải và sở hữu model của riêng mình
        detector = load_detector()
        video_stream = VideoStreamWidget(camera_src)
        if not video_stream.capture.isOpened():
            logging.error(f"[{window_name}] Không thể mở camera, luồng sẽ kết thúc: {camera_src}")
            return

        try:
            frame_count = -1
            last_results = None # Lưu kết quả xử lý gần nhất

            while not self.stop_event.is_set():
                ret, frame = video_stream.read()
                if not ret:
                    logging.warning(f"[{window_name}] Không thể đọc frame, luồng sẽ kết thúc.")
                    break
                
                frame_count += 1

                # Chỉ xử lý AI mỗi N frames
                if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                    # Sử dụng YOLOv8 để theo dõi đối tượng
                    # persist=True để duy trì track qua các frame
                    # tracker="bytetrack.yaml" để sử dụng ByteTrack
                    # classes=0 để chỉ phát hiện đối tượng 'person'
                    # conf=CONFIDENCE_THRESHOLD để model chỉ trả về các đối tượng đạt ngưỡng
                    results = detector.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, classes=0, conf=CONFIDENCE_THRESHOLD)
                    last_results = results # Cập nhật kết quả mới

                    # Logic lưu ảnh dựa trên sự xuất hiện của track ID mới
                    # Nếu có bất kỳ đối tượng nào được phát hiện, thử lưu ảnh
                    if results and len(results[0].boxes) > 0:
                        frame_to_save = results[0].plot() # Chuẩn bị frame để lưu
                        self.saver.save(frame_to_save, cam_id)

                # Luôn vẽ kết quả gần nhất lên frame hiển thị để video mượt
                if last_results:
                    display_frame = last_results[0].plot()
                else:
                    # Nếu chưa có kết quả nào (khi mới khởi động), hiển thị frame gốc
                    display_frame = frame

                cv2.imshow(window_name, display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    break
        finally:
            video_stream.release()
            cv2.destroyWindow(window_name)
            logging.info(f"Đã đóng luồng và cửa sổ cho {window_name}")

    def _supervisor_worker(self, sources):
        """Luồng giám sát, khởi động và khởi động lại các luồng camera khi cần."""
        logging.info("Luồng giám sát đã bắt đầu.")
        while not self.stop_event.is_set():
            for i, src in enumerate(sources):
                cam_id = i + 1
                window_name = f"Camera {cam_id}"
                thread = self.camera_threads.get(cam_id)
                if thread is None or not thread.is_alive():
                    if thread is not None: # Chỉ log nếu đây là khởi động lại
                        logging.warning(f"Phát hiện luồng Camera {cam_id} đã dừng. Đang khởi động lại...")
                    cam_thread = threading.Thread(target=self._camera_worker, args=(cam_id, src, window_name))
                    self.camera_threads[cam_id] = cam_thread
                    cam_thread.start()
            
            time.sleep(5) # Kiểm tra mỗi 5 giây
        logging.info("Luồng giám sát đã dừng.")

    def run(self):
        logging.info("Bắt đầu dịch vụ nhận diện... Nhấn 'q' trên bất kỳ cửa sổ nào để thoát.")
        sources = RTSP_URLS if USE_IP_CAMERA else [0]

        # Khởi tạo luồng giám sát camera
        supervisor_thread = threading.Thread(target=self._supervisor_worker, args=(sources,), daemon=True)
        supervisor_thread.start()
        self.supervisor_thread = supervisor_thread

        # Chờ sự kiện dừng (từ Ctrl+C hoặc nhấn 'q')
        self.stop_event.wait()

    def shutdown(self):
        """Dừng tất cả các luồng và giải phóng tài nguyên."""
        if not self.stop_event.is_set():
            self.stop_event.set()
        
        # Đảm bảo luồng giám sát và các luồng camera được join
        logging.info("Đang chờ các luồng kết thúc...")
        self.supervisor_thread.join(timeout=2)
        for thread in self.camera_threads.values():
            thread.join(timeout=2)

        logging.info("Tất cả các luồng đã đóng. Kết thúc chương trình.")
        cv2.destroyAllWindows()
