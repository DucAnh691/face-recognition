from ultralytics import YOLO
import logging

def load_detector():
    """Tải model YOLOv8."""
    # yolov8s.pt (small) có độ chính xác cao hơn yolov8n.pt (nano)
    # mà vẫn giữ hiệu suất tốt trên CPU.
    model_name = 'yolov8s.pt'
    logging.info(f"Đang tải model: {model_name}")
    model = YOLO(model_name)
    logging.info("Model YOLOv8 đã tải xong.")
    return model
