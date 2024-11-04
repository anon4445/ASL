# Client Server Based Real Time Sign Language Detection with Websockets and Yolov5

A FastAPI-based application for real-time American Sign Language Recognition (ASLR) using a custom YOLOv5 model. This API allows users to process real-time video feeds from a webcam or upload images to detect and recognize ASL gestures.

## Features

- **Real-time Object Detection**: Uses YOLOv5 to detect ASL gestures in a live video feed.
- **WebSocket Support**: Streams video frames with detected objects via WebSocket.
- **Image Upload Prediction**: Allows users to upload an image for ASL prediction.
- **Cross-Origin Resource Sharing (CORS)**: Supports cross-origin requests for easy integration with web applications.
- **Static File Support**: Serves static files, including a main HTML file (`index.html`), for the front-end interface.

## Requirements

To run this application, install the necessary dependencies listed in `requirements.txt`:

```plaintext
# YOLOv5 requirements
gitpython>=3.1.30
matplotlib>=3.3
numpy>=1.23.5
opencv-python>=4.1.1
pillow>=10.3.0
psutil
PyYAML>=5.3.1
requests>=2.32.2
scipy>=1.4.1
thop>=0.1.1
torch>=1.8.0
torchvision>=0.9.0
tqdm>=4.66.3
ultralytics>=8.2.34
pandas>=1.1.4
seaborn>=0.11.0
