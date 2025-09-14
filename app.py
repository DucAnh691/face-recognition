from core.service import FaceRecognitionService
import logging

if __name__ == "__main__":
    service = FaceRecognitionService()
    try:
        service.run()
    finally:
        service.shutdown()
