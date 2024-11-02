import cv2
import torch
import base64
from PIL import Image
from fastapi import FastAPI, WebSocket
from fastapi import FastAPI, UploadFile, File 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
import io
from fastapi.staticfiles import StaticFiles 
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='aslr.pt', source='local')

app = FastAPI(title="ASL Prediction", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static") 

@app.get("/")
def serve_homepage():
    return FileResponse('static/index.html') 
# WebSocket endpoint for real-time video feed
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Open webcam (source 0)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv5 Inference
        results = model(frame)

        # Draw bounding boxes and labels on the frame
        labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        for i in range(len(labels)):
            row = coords[i]
            if row[4] >= 0.2:  # Confidence threshold
                x1, y1, x2, y2 = int(row[0] * frame.shape[1]), int(row[1] * frame.shape[0]), int(row[2] * frame.shape[1]), int(row[3] * frame.shape[0])
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                frame = cv2.putText(frame, model.names[int(labels[i])], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Encode frame as JPEG and send via WebSocket
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        await websocket.send_text(jpg_as_text)

    cap.release()

def predict_image(image: Image.Image):
    results = model(image)  
    return results.pandas().xyxy[0].to_dict(orient="records")  

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()  # Read the uploaded file as bytes
    image = Image.open(io.BytesIO(contents))  # Open the image with PIL
    predictions = predict_image(image)  # Get predictions from the model
    return JSONResponse(content={"predictions": predictions})  # Return the predictions as JSON