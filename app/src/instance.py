import base64
import random
import tempfile

import cv2 as cv
from fastapi import UploadFile
import numpy as np


MODEL_PATH = './ml-models/mask-rcnn-coco/'
MAX_SIZE = 10000
MAX_SIZE_MODEL = 1024


def draw_masks(image, boxes, masks, conf_threshold=0.5, mask_threshold=0.3):
    number_of_objects = boxes.shape[2]
    for i in range(number_of_objects):
        box = boxes[0, 0, i]
        conf = box[2]
        if conf > conf_threshold:
            class_id = int(box[1])
            left, top, right, bottom = get_box_edges(image, box)
            color = get_color(class_id)

            # Extract and resize the mask for the object
            mask = masks[i, class_id]
            mask = cv.resize(mask, (right - left + 1, bottom - top + 1))
            mask = (mask > mask_threshold)

            # Overlay mask
            overlay = 0.6
            roi = image[top:bottom+1, left:right+1][mask]
            image[top:bottom+1, left:right+1][mask] = (
                    overlay * color + (1 - overlay) * roi).astype(np.uint8)

            # Draw contours
            contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_TREE,
                                          cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(
                image[top:bottom+1, left:right+1], contours, -1,
                (int(color[0]), int(color[1]), int(color[2])), 2,
            )


def draw_boxes(image, boxes, conf_threshold=0.5):
    number_of_objects = boxes.shape[2]
    for i in range(number_of_objects):
        box = boxes[0, 0, i]
        conf = box[2]
        if conf > conf_threshold:
            left, top, right, bottom = get_box_edges(image, box)

            # Draw a bounding box
            cv.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)


def draw_confs(image, boxes, conf_threshold=0.5):
    number_of_objects = boxes.shape[2]
    for i in range(number_of_objects):
        box = boxes[0, 0, i]
        conf = box[2]
        if conf > conf_threshold:
            class_id = int(box[1])
            left, top, right, bottom = get_box_edges(image, box)

            # Display the label at the top of the bounding box
            label = f'{classes[class_id]}: {int(100*conf)} %'
            cv.putText(image, label, (left+2, top+20), cv.FONT_HERSHEY_SIMPLEX,
                       0.75, (255, 255, 255), 2)


def get_box_edges(image, box):
    height = image.shape[0]
    width = image.shape[1]

    # Extract the box edges
    left = int(width * box[3])
    top = int(height * box[4])
    right = int(width * box[5])
    bottom = int(height * box[6])

    # Limit the box edges
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(0, min(right, width - 1))
    bottom = max(0, min(bottom, height - 1))

    return left, top, right, bottom


def get_color(class_id, color_per_class=False):
    if color_per_class:
        color = object_colors[class_id % len(object_colors)]
    else:
        color_index = random.randint(0, len(object_colors) - 1)
        color = object_colors[color_index]

    return color


def get_response_data(file: UploadFile) -> dict:
    # Convert UploadFile to OpenCV image
    file_array = np.fromfile(file.file, np.uint8)
    if len(file_array) == 0:  # empty file check
        msg = 'Choose a file.'
        return {'info': msg}
    img = cv.imdecode(file_array, cv.IMREAD_COLOR)
    if img is None:  # operation status check
        msg = 'Image must be in JPEG or PNG format. Choose another file.'
        return {'info': msg}

    # Check for maximum image size
    height, width = img.shape[:2]
    if height > MAX_SIZE or width > MAX_SIZE:
        msg = 'Image width and height must be less than 5000px.'\
              + ' Choose another file.'
        return {'info': msg}

    # Resize image for better Mask R-CNN performance
    if max(width, height) > MAX_SIZE_MODEL:
        aspect_ratio = width / height
        if aspect_ratio > 1.0:
            shape = (MAX_SIZE_MODEL, int(MAX_SIZE_MODEL / aspect_ratio))
        else:
            shape = (int(MAX_SIZE_MODEL * aspect_ratio), MAX_SIZE_MODEL)
        img = cv.resize(img, shape, interpolation=cv.INTER_AREA)

    # Apply the model
    try:
        blob = cv.dnn.blobFromImage(img, swapRB=True, crop=False)
        model.setInput(blob)
        boxes, masks = model.forward(['detection_out_final',
                                      'detection_masks'])
    except Exception as err:
        err_type = 'Exception'
        print(f'{__name__, err_type}: {err}')
        return {'info': f'{err_type}: {err}'}

    # Draw masks, boxes and confidences for each of the detected objects
    draw_masks(img, boxes, masks)
    draw_boxes(img, boxes)
    draw_confs(img, boxes)

    # Convert OpenCV image to base64 string
    status, file_array = cv.imencode('.jpg', img)
    if not status:
        err_type = 'Error'
        err_msg = "OpenCV can't encode image"
        print(f'{__name__, err_type}: {err_msg}')
        return {'info': f'{err_type}: {err_msg}'}
    with tempfile.SpooledTemporaryFile() as fp:
        file_array.tofile(fp)
        fp.seek(0)
        bytes_array = base64.b64encode(fp.read())
    image_base64 = 'data:image/jpeg;base64,'
    try:
        image_base64 += bytes_array.decode()
    except UnicodeError as err:
        err_type = 'UnicodeError'
        print(f'{__name__, err_type}: {err}')
        return {'info': f'{err_type}: {err}'}

    # Form the info string
    t, _ = model.getPerfProfile()
    info = 'Done! Inference time: '\
           + f'{int(t * 1000.0 / cv.getTickFrequency())} ms.'

    return {'image': image_base64, 'info': info}


# Load the model
model = cv.dnn.readNetFromTensorflow(
    MODEL_PATH + 'frozen_inference_graph.pb',
    MODEL_PATH + 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt',
)
model.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Load names of classes
with open(MODEL_PATH + 'labels.txt', 'r') as f:
    classes = f.read().split('\n')

# Define colors for masks
object_colors = np.random.randint(0, 255, (len(classes), 3))
