from ultralytics import YOLO
import cv2 as cv
import numpy as np
import tensorflow as tf

THRESHOLD = 0.5  # Set your desired confidence threshold
NMS_THRESHOLD = 0.5 # Intersection over Union threshold for Non Maximum Suppresion for bounding boxes
all_frames_output = []

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
def getNmsIndicies(detected_object_list, nms_threshold):
    bboxes      = [list(detected_object["bbox"].values()) for detected_object in detected_object_list]
    scores  = [detected_object["confidence"] for detected_object in detected_object_list]

    indicies = cv.dnn.NMSBoxes(bboxes, scores, score_threshold=0.0, nms_threshold=nms_threshold)
    return indicies

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

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
    boxes = boxes.T
    single_frame_output = []

    for i in range(class_probs.shape[0]):
        class_id = np.argmax(class_probs[i])
        confidence = class_probs[i][class_id]
  
        if confidence > THRESHOLD:
            detected_object = {
                "class" : class_names[class_id], 
                "confidence": float(confidence),
                "bbox" : {
                    # X center, Y center
                    "x": float(boxes[class_id][0]),
                    "y": float(boxes[class_id][1]),
                    "w": float(boxes[class_id][2]),
                    "h": float(boxes[class_id][3]),
                }
            }
            single_frame_output.append(detected_object)

    if len(single_frame_output) > 0:
        # Perform non maximum suppresion over 8400 predictions to remove overlaps
        indicies = getNmsIndicies(single_frame_output, NMS_THRESHOLD)
        filtered_output = [single_frame_output[i] for i in indicies]

        all_frames_output.append(filtered_output)
          # print(filtered_output)


    cv.imshow("Video", frame)

    if cv.waitKey(1) == ord('q'):
        break

print(all_frames_output)
cap.release()
cv.destroyAllWindows()
