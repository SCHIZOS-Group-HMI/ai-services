from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import Request
import base64
from PIL import Image
import io
import numpy as np

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
    
class audio(BaseModel):
    pass    

@asynccontextmanager
async def lifespan(app: FastAPI):
    from service.object_detection.YoloService import YoloService
    from service.audio_classification.YamnetService import YamnetService
    app.state.yolo_model = YoloService(0.8, 0.5)
    app.state.yamnet_model = YamnetService(0.5)
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"msg" : "succesfully called api"}

# Only send data using POST
@app.post("/scan")
async def scanEnvironment(request: Request, scan_data: ScanData):
    # decode image here
    image_bytes = base64.b64decode(scan_data.image)
    decoded_image = np.array(Image.open(io.BytesIO(image_bytes)))

    obj_model = request.app.state.yolo_model
    obj_result = obj_model.detect(decoded_image)

    # 16-PCM format is already a waveform so only need to decode the encoded waveform to get waveform
    waveform = base64.b64decode(scan_data.audio)
    audio_model = request.app.state.yamnet_model
    audio_result = audio_model.detect(waveform)

    result = {
        "object_detection": obj_result,
        "audio_detection" : audio_result
    }
    
    return result