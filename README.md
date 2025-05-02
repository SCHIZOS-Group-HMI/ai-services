# AI Services API

This is an API service that offers object detection and audio classification using a pre-trained model. It processes image and audio data to identify objects and detect audio patterns.

## 1. Installation

Before installing the dependencies, it is recommended to create a virtual environment to manage dependencies.

### Create a Virtual Environment

Run the following command to create a virtual environment:

```bash
python -m venv hmi_venv
```

### Activate the Virtual Environment
* On windows 
```bash
hmi_venv\Scripts\activate
```

### Model Download
To use the Yamnet model for audio classification, you'll need to download the pre-trained model. Follow these steps:

1. Download the **Yamnet TensorFlow 2 model** from this link:  
   [Yamnet TensorFlow 2 Model on Kaggle](https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1?select=variables)

2. After downloading the file `yamnet-tensorflow2-yamnet-v1.tar.gz`, extract the file into the following directory in your project: **service/audio_classification/yamnet_model**

3. Ensure that this directory contains the model files including `saved_model.pb`, `variables/`, and other assets required for the model.

## 2. Running the Server
To run the FastAPI server, use the following command:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```
This will start the server locally. Once running, you can access the API at the typical FastAPI URL, usually http://127.0.0.1:8000, and the documentation will be available at http://127.0.0.1:8000/docs.

## 3. Endpoints and HTTP Requests
### POST: /scan

This endpoint processes a scan request, which includes image and audio data along with metadata.

#### Request Body

The request body should be a JSON object with the following structure:

```json
{
  "timestamp": "2025-04-25T12:00:00Z",
  "image": "base64_encoded_image_string",
  "audio": "base64_encoded_audio_data",
  "audio_amplitude": 123.45,
  "metadata": {
    "sample_rate": 44100,
    "channels": 1,
    "audio_format": "PCM_16BIT",
    "image_format": "JPEG",
    "resolution": "1280x720"
  }
}
```

### Fields Explanation

- **timestamp**: A timestamp in ISO 8601 format.
- **image**: A base64-encoded string representing the image data.
- **audio**: A base64-encoded string representing the audio data.
- **audio_amplitude**: A float representing the amplitude of the audio signal.
- **metadata**: Additional metadata for the audio and image, including:
  - **sample_rate**: The audio sample rate (e.g., 44100).
  - **channels**: Number of audio channels (e.g., 1 for mono, 2 for stereo).
  - **audio_format**: The audio format (e.g., "PCM_16BIT").
  - **image_format**: The image format (e.g., "JPEG").
  - **resolution**: The resolution of the image (e.g., "1280x720").

#### Response
```json
{
  "object_detection": [
    {
      "class": "person",
      "confidence": 0.8987748026847839,
      "bbox": {
        "x": 0.006538123358041048,
        "y": 0.0066288490779697895,
        "w": 0.013650982640683651,
        "h": 0.012226744554936886
      }
    }
  ],
  "audio_detection": {
    "threshold": 0.5,
    "results": [
      {
        "class": "speech",
        "confidence": 0.92
      }
    ]
  }
}
```
#### Field explanation:
- **object_detection**: An array containing detected objects in the image, each with:
  - **class**: The class of the detected object (e.g., "person").
  - **confidence**: The confidence score of the detection.
  - **bbox**: The bounding box for the detected object, with **x**, **y**, **w** (width), and **h** (height).
- **audio_detection**: The results of the audio classification, containing:
  - **threshold**: The detection threshold used for classification.
  - **results**: An array with detected audio classes and their confidence scores (e.g., "speech" with confidence 0.92).
