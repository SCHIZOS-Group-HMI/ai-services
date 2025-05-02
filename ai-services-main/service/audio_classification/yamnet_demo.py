
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import io
import sounddevice as sd

model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')

# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_map_csv = io.StringIO(class_map_csv_text)
  class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
  class_names = class_names[1:]  # Skip CSV header

  return class_names

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(tf.io.read_file(class_map_path).numpy().decode('utf-8'))

SAMPLE_RATE = 16000 # samples per second 
SAMPLE_TIME = 0.2
CHUNK_SAMPLES = int(SAMPLE_RATE * SAMPLE_TIME)

def callback(indata, frames, time, status):
    mono_audio = indata[:, 0]
    if len(mono_audio) >= CHUNK_SAMPLES:
        waveform = mono_audio[:CHUNK_SAMPLES]

        # Audio classification
        scores, embeddings, log_mel_spectrogram = model(waveform)

        scores.shape.assert_is_compatible_with([None, 521]) # 512 classes
        embeddings.shape.assert_is_compatible_with([None, 1024])
        log_mel_spectrogram.shape.assert_is_compatible_with([None, 64])
        
        scores = scores.numpy().mean(axis=0)
        THRESHOLD = 0.5

        result = {}
        for i in range(len(scores)):
            if scores[i] >= THRESHOLD:
                result[class_names[i]] = scores[i]
        
        print(f"Pred '{class_names[scores.argmax()]}' Results greater than {THRESHOLD} : {result}")
        # print(class_names[scores.numpy().mean(axis=0).argmax()]) # (1, 512).mean() = (512, )

stream = sd.InputStream(channels=1, callback=callback, samplerate=SAMPLE_RATE, blocksize=CHUNK_SAMPLES)
with stream:
    input("Press Enter to stop...\n")
