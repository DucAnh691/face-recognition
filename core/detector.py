import torch
import logging
from transformers import pipeline

def load_detector():
    """Load model phát hiện đối tượng (DETR)."""
    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"Phát hiện GPU: {gpu_name}, sẽ chạy trên GPU.")
    else:
        device = -1
        logging.info("Không có GPU, sẽ chạy trên CPU.")

    return pipeline(
        "object-detection",
        model="facebook/detr-resnet-101",
        device=device
    )
