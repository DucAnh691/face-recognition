import cv2

def detect_motion(previous_gray, current_gray, threshold=30):
    """Kiểm tra có chuyển động giữa 2 frame hay không."""
    frame_delta = cv2.absdiff(previous_gray, current_gray)
    _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
    motion_detected = thresh.sum() > 0
    return motion_detected, thresh
