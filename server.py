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
    app.state.yolo_model = YoloService(0.8, 0.5)
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

    model = request.app.state.yolo_model
    result = model.detect(decoded_image)


    return result