import tensorflow as tf
import csv
import io
from functools import partial
import os

class YamnetService:
    def __init__(self, THRESHOLD=0.5):
        model_path = os.path.join(os.path.dirname(__file__), "yamnet_model")
        print(model_path)
        self.model = tf.saved_model.load(model_path)
        class_map_path = self.model.class_map_path().numpy()
        self.CLASS_NAMES = self.CLASS_NAMES_from_csv(tf.io.read_file(class_map_path).numpy().decode('utf-8'))
        self.THRESHOLD=THRESHOLD

    def CLASS_NAMES_from_csv(self, class_map_csv_text):
        """Returns list of class names corresponding to score vector."""
        class_map_csv = io.StringIO(class_map_csv_text)
        class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
        class_names = class_names[1:]  # Skip CSV header

        return class_names
    
    def preprocess(self, waveform=None):
        # If amp too small then amp it up
        pass
        
    def run_inference(self, waveform):
        scores, embeddings, log_mel_spectrogram = self.model(waveform)
        scores.shape.assert_is_compatible_with([None, 521])
        embeddings.shape.assert_is_compatible_with([None, 1024])
        log_mel_spectrogram.shape.assert_is_compatible_with([None, 64])
        return scores

    def postprocess(self, scores):
        # Lấy tensor sang numpy và tính trung bình mỗi class
        final_scores = scores.numpy().mean(axis=0)

        # Lọc ra những class >= threshold và cast về Python float
        result = {}
        for i, score in enumerate(final_scores):
            if score >= self.THRESHOLD:
                result[self.CLASS_NAMES[i]] = float(score)

        # Trả về dict hoàn toàn với Python native types
        return {
            "threshold": float(self.THRESHOLD),
            "results": result
        }

    def detect(self, waveform):
        scores = self.run_inference(waveform)
        return self.postprocess(scores)

# import sounddevice as sd

# def callback(indata, frames, time, status, prediction_model):
#     mono_audio = indata[:, 0]
#     if len(mono_audio) >= CHUNK_SAMPLES:
#         waveform = mono_audio[:CHUNK_SAMPLES]

#         # Audio classification
#         result = prediction_model.detect(waveform)
#         print(result)

# if __name__ == "__main__":
#     service = YamnetService()
#     SAMPLE_RATE = 16000 # bits per second
#     SAMPLE_TIME = 0.2
#     CHUNK_SAMPLES = int(SAMPLE_RATE * SAMPLE_TIME)

#     callback_with_model = partial(callback, prediction_model=service)
#     # the code above is shorter and equivalent to below
#     # def callback_with_model(indata, frames, time, status):
#     #     callback(indata, frames, time, status, service)
    
#     stream = sd.InputStream(channels=1, callback=callback_with_model, samplerate=SAMPLE_RATE, blocksize=CHUNK_SAMPLES)
#     with stream:
#         input("Press Enter to stop...\n")
