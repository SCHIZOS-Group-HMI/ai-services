from ultralytics import YOLO
import cv2 as cv
import numpy as np
import tensorflow as tf
import os

class YoloService:
    CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    '''
    Perform non maximum suppresion for multiple bounding boxes detected of many overlapping predictions.

    Parameters
    ----------
    detected_object: list
        A list of object/dictionary that contains the information of detected objects in a single frame.
    
    nms_threshold: float
        The threshold for non maximum suppresion.
    
    Returns
    -------
    list
        A list of indicies of bounding boxes that are kept after non maximum suppresions.
    '''
    def getNmsIndicies(self, detected_object_list, nms_threshold):
        bboxes      = [list(detected_object["bbox"].values()) for detected_object in detected_object_list]
        scores  = [detected_object["confidence"] for detected_object in detected_object_list]

        indicies = cv.dnn.NMSBoxes(bboxes, scores, score_threshold=0.0, nms_threshold=nms_threshold)
        return indicies

        
    def __init__(self, DETECTION_THRESHOLD=0.5, NMS_THRESHOLD=0.5):
        self.interpreter = tf.lite.Interpreter(model_path='service/object_detection/yolo11n_float32.tflite')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        _, self.input_height, self.input_width, _ = self.input_shape

        self.DETECTION_THRESHOLD = DETECTION_THRESHOLD
        self.NMS_THRESHOLD = NMS_THRESHOLD
    
    def preprocess(self, frame):
        resized_frame = cv.resize(frame, (self.input_width, self.input_height))
        model_input = resized_frame.astype(np.float32) / 255.0
        # Add batch dimension
        model_input = np.expand_dims(model_input, axis=0)
        return model_input
    
    def run_inference(self, model_input):
        # model_input = self.preprocess(frame)

        # 'index' correspond to the position of input tensor in the model
        self.interpreter.set_tensor(self.input_details[0]['index'], model_input)
        self.interpreter.invoke()

        model_output = self.interpreter.get_tensor(self.output_details[0]['index']) # shape: [1, 84, 8400]
        return model_output
    
    def postprocess(self, model_output):
        bboxes      = model_output[0, 0:4, :]
        class_probs = model_output[0, 4:, :]

        # Transpose to [8400, 80]
        bboxes      = bboxes.T
        class_probs = class_probs.T

        single_frame_output = []

        for i in range(class_probs.shape[0]):
            class_id = np.argmax(class_probs[i])
            confidence = class_probs[i][class_id]

            if confidence > self.DETECTION_THRESHOLD:
                detected_object = {
                    "class" : self.CLASS_NAMES[class_id], 
                    "confidence": float(confidence),
                    "bbox" : {
                        # X center, Y center
                        "x": float(bboxes[class_id][0]),
                        "y": float(bboxes[class_id][1]),
                        "w": float(bboxes[class_id][2]),
                        "h": float(bboxes[class_id][3]),
                    }
                }
                single_frame_output.append(detected_object)

        if len(single_frame_output) > 0:
            # Perform non maximum suppresion over 8400 predictions to remove overlaps
            indicies = self.getNmsIndicies(single_frame_output, self.NMS_THRESHOLD)
            filtered_output = [single_frame_output[i] for i in indicies]
            return filtered_output
        
        return None
    
    # This will be used for API endpoint
    def detect(self, frame):
        model_input = self.preprocess(frame)
        model_output = self.run_inference(model_input)
        return self.postprocess(model_output)
    
# if __name__ == "__main__":
#     detector = YoloService(0.8, 0.5)
#     cap = cv.VideoCapture(0)

#     if not cap.isOpened():
#         print("Cannot open camera")
#         exit()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Cannot get frame")
#             break

#         result = detector.detect(frame)

#         print(result)

#         cv.imshow("Video", frame)

#         if cv.waitKey(1) == ord('q'):
#             break
    
#     cap.release()
#     cv.destroyAllWindows()