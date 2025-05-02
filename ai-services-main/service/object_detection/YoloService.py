import cv2 as cv
import numpy as np
import tensorflow as tf

class YoloService:
    # 80 class COCO, bỏ bớt ở đây cho gọn
    CLASS_NAMES = [
        'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
        'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
        'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
        'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
        'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
        'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
        'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
        'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote',
        'keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book',
        'clock','vase','scissors','teddy bear','hair drier','toothbrush'
    ]

    def __init__(self,
                 DETECTION_THRESHOLD: float = 0.3,  # giảm ngưỡng xuống 0.3
                 NMS_THRESHOLD: float = 0.5):
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(
            model_path='service/object_detection/yolo11n_float32.tflite'
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        _, self.input_height, self.input_width, _ = self.input_details[0]['shape']

        self.DETECTION_THRESHOLD = DETECTION_THRESHOLD
        self.NMS_THRESHOLD = NMS_THRESHOLD

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Frame đã là BGR, resize, normalize
        resized = cv.resize(frame, (self.input_width, self.input_height))
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)  # [1, H, W, 3]

    def run_inference(self, model_input: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self.input_details[0]['index'], model_input)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])  # [1, 84, 8400]

    def getNmsIndicies(self, detected_list, nms_threshold):
        bboxes = [list(obj["bbox"].values()) for obj in detected_list]
        scores = [obj["confidence"] for obj in detected_list]
        indices = cv.dnn.NMSBoxes(bboxes, scores, score_threshold=0.0, nms_threshold=nms_threshold)
        return [i[0] if isinstance(i, (list, tuple)) else i for i in indices]

    def postprocess(self, model_output: np.ndarray):
        # tách boxes và probs rồi transpose
        boxes = model_output[0, 0:4, :].T      # [8400, 4]
        probs = model_output[0, 4:, :].T       # [8400, 80]

        detections = []
        for i in range(probs.shape[0]):
            class_id = int(np.argmax(probs[i]))
            confidence = float(probs[i][class_id])
            if confidence > self.DETECTION_THRESHOLD:
                x, y, w, h = boxes[i]  # SỬA: dùng boxes[i]
                detections.append({
                    "class": self.CLASS_NAMES[class_id],
                    "confidence": confidence,
                    "bbox": {
                        "x": float(x),
                        "y": float(y),
                        "w": float(w),
                        "h": float(h)
                    }
                })

        if not detections:
            return None

        keep = self.getNmsIndicies(detections, self.NMS_THRESHOLD)
        filtered = [detections[i] for i in keep]
        return filtered

    def detect(self, frame: np.ndarray):
        inp = self.preprocess(frame)
        out = self.run_inference(inp)
        return self.postprocess(out)
