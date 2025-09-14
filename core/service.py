import cv2
import logging
from PIL import Image
import time
import threading
from queue import Queue, Empty, Full

from config import (
    RTSP_URLS, USE_IP_CAMERA, PROCESS_EVERY_N_FRAMES, CONFIDENCE_THRESHOLD,
    MOTION_THRESHOLD, LOG_FILE, BATCH_SIZE, MAX_QUEUE_SIZE
)
from core.camera import VideoStreamWidget
from core.detector import load_detector
from core.motion import detect_motion
from core.saver import CaptureSaver
from core.utils import setup_logging, resize_frame, draw_detection

class FaceRecognitionService:
    def __init__(self):
        setup_logging(LOG_FILE)
        self.detector = load_detector()
        self.saver = CaptureSaver()
        self.frame_queues = {}
        self.results_queues = {}
        self.camera_threads = {}
        self.threads = []
        self.stop_event = threading.Event()

    def _inference_worker(self):
        """Luồng chuyên xử lý AI, nhận frame từ hàng đợi và xử lý theo lô."""
        logging.info("Luồng xử lý AI đã bắt đầu.")
        cam_ids = list(self.frame_queues.keys())
        
        while not self.stop_event.is_set():
            batch = []
            # Thu thập frame từ mỗi camera một cách công bằng (round-robin)
            for cam_id in cam_ids:
                try:
                    item = self.frame_queues[cam_id].get_nowait()
                    batch.append(item)
                    if len(batch) >= BATCH_SIZE:
                        break
                except Empty:
                    continue

            if not batch:
                time.sleep(0.01) # Hàng đợi trống, nghỉ một chút để tránh busy-waiting
                continue

            # Tạo một generator để pipeline xử lý hiệu quả
            def image_generator(batch_data):
                for _, frame, _ in batch_data:
                    yield Image.fromarray(cv2.cvtColor(resize_frame(frame), cv2.COLOR_BGR2RGB))

            # Phân phối kết quả về các hàng đợi tương ứng
            all_detections = list(self.detector(image_generator(batch)))
            for i, (cam_id, _, _) in enumerate(batch):
                try:
                    self.results_queues[cam_id].put(all_detections[i], block=False)
                except Full:
                    logging.warning(f"Hàng đợi kết quả của Camera {cam_id} bị đầy. Bỏ qua kết quả.")
        logging.info("Luồng xử lý AI đã dừng.")

    def _camera_worker(self, cam_id, camera_src, window_name):
        """Luồng chuyên đọc frame từ camera, hiển thị và xử lý logic phụ."""
        video_stream = VideoStreamWidget(camera_src)
        if not video_stream.capture.isOpened():
            logging.error(f"[{window_name}] Không thể mở camera, luồng sẽ kết thúc: {camera_src}")
            return

        try:
            previous_gray, frame_count = None, 0
            detections = []

            while not self.stop_event.is_set():
                ret, frame = video_stream.read()
                if not ret:
                    logging.warning(f"[{window_name}] Không thể đọc frame, luồng sẽ kết thúc.")
                    break

                display_frame = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                # Gửi frame đến luồng AI để xử lý
                frame_count += 1
                if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                    try:
                        self.frame_queues[cam_id].put((cam_id, frame, gray), block=False)
                    except Full:
                        # Hàng đợi đầy, nghỉ một chút để luồng AI có thời gian xử lý
                        # và tránh spam log quá nhiều.
                        time.sleep(0.1)
                        # logging.warning(f"[{window_name}] Hàng đợi frame bị đầy. Bỏ qua frame.")

                # Nhận kết quả từ luồng AI (nếu có)
                try:
                    detections = self.results_queues[cam_id].get(block=False)
                except Empty:
                    pass # Không có kết quả mới, tiếp tục dùng kết quả cũ

                motion_detected, thresh = False, None
                if previous_gray is not None:
                    motion_detected, thresh = detect_motion(previous_gray, gray, MOTION_THRESHOLD)

                person_is_moving = False
                if detections:
                    for detection in detections:
                        box, label, score = detection["box"], detection["label"], detection["score"]
                        if label == "person" and score > CONFIDENCE_THRESHOLD:
                            draw_detection(display_frame, box, label, score)
                            
                            # Tối ưu hóa: Chỉ kiểm tra chuyển động bên trong vùng có người
                            if motion_detected:
                                xmin, ymin, xmax, ymax = map(int, (box["xmin"], box["ymin"], box["xmax"], box["ymax"]))
                                # Cắt vùng chuyển động tương ứng với bounding box của người
                                roi_motion_mask = thresh[ymin:ymax, xmin:xmax]
                                
                                # Kiểm tra xem có đủ pixel chuyển động trong vùng ROI không
                                if roi_motion_mask.size > 0 and cv2.countNonZero(roi_motion_mask) > MOTION_THRESHOLD:
                                    person_is_moving = True

                if person_is_moving:
                    self.saver.save(display_frame)

                cv2.imshow(window_name, display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    break

                previous_gray = gray

        finally:
            video_stream.release()
            cv2.destroyWindow(window_name)
            logging.info(f"Đã đóng luồng và cửa sổ cho {window_name}")

    def _supervisor_worker(self, sources):
        """Luồng giám sát và khởi động lại các luồng camera bị lỗi."""
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

        # Khởi tạo luồng xử lý AI
        inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.threads.append(inference_thread)
        inference_thread.start()

        # Khởi tạo hàng đợi kết quả cho mỗi camera
        for i, src in enumerate(sources):
            cam_id = i + 1
            self.frame_queues[cam_id] = Queue(maxsize=MAX_QUEUE_SIZE)
            self.results_queues[cam_id] = Queue(maxsize=MAX_QUEUE_SIZE)

        # Khởi tạo luồng giám sát camera
        supervisor_thread = threading.Thread(target=self._supervisor_worker, args=(sources,), daemon=True)
        self.threads.append(supervisor_thread)
        supervisor_thread.start()

        # Chờ sự kiện dừng (từ Ctrl+C hoặc nhấn 'q')
        self.stop_event.wait()

    def shutdown(self):
        """Dừng tất cả các luồng và giải phóng tài nguyên."""
        if not self.stop_event.is_set():
            self.stop_event.set()
        
        # Dừng tất cả các luồng khác
        logging.info("Đang chờ các luồng còn lại kết thúc...")
        for thread in self.threads:
            thread.join(timeout=2)
        
        # Đảm bảo các luồng camera cũng được join
        for thread in self.camera_threads.values():
            thread.join(timeout=2)

        logging.info("Tất cả các luồng đã đóng. Kết thúc chương trình.")
        cv2.destroyAllWindows()
