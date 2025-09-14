# Hugging Face Video Face Recognition

This project uses Hugging Face's `transformers` library to perform **face detection** on videos. It processes a video file, detects faces in each frame using the `facebook/detr-resnet-50` model, and saves a new video with bounding boxes drawn around the detected faces.

## Project Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd huggingface-video-face-recognition
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

Run the script from your terminal, providing the path to your video file as an argument. The output video will be saved in the same directory with `_output` appended to the original filename.

```bash
python app.py path/to/your/video.mp4
```
