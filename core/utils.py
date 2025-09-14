import cv2
import logging
import numpy as np
from config import SCALE_FACTOR, CONFIDENCE_THRESHOLD

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def resize_frame(frame):
    return cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

def draw_detection(frame, box, label, score):
    xmin, ymin, xmax, ymax = map(int, (box["xmin"], box["ymin"], box["xmax"], box["ymax"]))
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.putText(frame, f"{label}: {score:.2f}", (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
