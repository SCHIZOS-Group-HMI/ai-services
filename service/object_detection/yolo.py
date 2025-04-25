from ultralytics import YOLO
import cv2 as cv
import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='yolo11n_float32.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
_, input_height, input_width, _ = input_shape

# Class labels (COCO)
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
               'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
               'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
               'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
               'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

THRESHOLD = 0.5  # Set your desired confidence threshold

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't get frame")
        break

    frame = cv.flip(frame, 1)
    resized_frame = cv.resize(frame, (input_width, input_height))
    model_input = resized_frame.astype(np.float32) / 255.0
    model_input = np.expand_dims(model_input, axis=0)

    interpreter.set_tensor(input_details[0]['index'], model_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])  # shape: [1, 84, 8400]

    boxes = output_data[0, 0:4, :]       # shape: [4, 8400]
    class_probs = output_data[0, 4:, :]  # shape: [80, 8400]

    # Transpose to [8400, 80]
    class_probs = class_probs.T

    for i in range(class_probs.shape[0]):
        class_id = np.argmax(class_probs[i])
        confidence = class_probs[i][class_id]

        if confidence > THRESHOLD:
            print(f"{class_names[class_id]}: {confidence:.2f}", end=", ")

    print()
    cv.imshow("Video", frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
