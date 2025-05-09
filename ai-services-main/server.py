import io
import base64
import numpy as np
from PIL import Image
import librosa
import cv2 as cv
import tensorflow as tf

from fastapi import FastAPI, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from starlette.responses import JSONResponse
import json

class MetaData(BaseModel):
    sample_rate: int
    channels: int
    audio_format: str
    image_format: str
    resolution: str

class ScanData(BaseModel):
    timestamp: str
    image: str
    audio: str
    audio_amplitude: float
    metadata: MetaData

@asynccontextmanager
async def lifespan(app: FastAPI):
    from service.object_detection.YoloService import YoloService
    from service.audio_classification.YamnetService import YamnetService
    from service.chatbot.GeminiService import GeminiService
    # Giảm threshold YOLO xuống 0.3
    app.state.yolo_model = YoloService(0.3, 0.5)
    app.state.yamnet_model = YamnetService(0.3)
    app.state.gemini_model = GeminiService()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"msg": "successfully called api"}

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    print(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"message": "Validation error", "detail": str(exc)}
    )

@app.post("/scan")
async def scanEnvironment(request: Request, scan_data: ScanData):
    # --- xử lý ảnh ---
    image_bytes = base64.b64decode(scan_data.image)
    decoded_image = np.array(Image.open(io.BytesIO(image_bytes)))
    # Chuyển RGB→BGR để khớp OpenCV
    decoded_image = cv.cvtColor(decoded_image, cv.COLOR_RGB2BGR)

    obj_result = request.app.state.yolo_model.detect(decoded_image)

    # --- xử lý âm thanh ---
    audio_bytes = base64.b64decode(scan_data.audio)
    waveform, _ = librosa.load(
        io.BytesIO(audio_bytes),
        sr=16000,   # resample về 16 kHz
        mono=True,
        dtype=np.float32
    )
    audio_result = request.app.state.yamnet_model.detect(waveform)

    return {
        "object_detection": obj_result,
        "audio_detection": audio_result
    }


@app.post("/chat")
async def getBotResponse(request: Request, inputData: dict):
    inputDataString = json.dumps(inputData)
    chat_response = request.app.state.gemini_model.getResponse(inputDataString)
    return {
        "chat_response": chat_response
    }
