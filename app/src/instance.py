import random

import cv2 as cv
import numpy as np


MODEL_PATH = './ml-models/mask-rcnn-coco/'


# Load model
model = cv.dnn.readNetFromTensorflow(
    MODEL_PATH + 'frozen_inference_graph.pb',
    MODEL_PATH + 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt',
)
# model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # this code is from learnopencv.com. Should I use it?
# model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # this code is from learnopencv.com. Should I use it?

# Load names of classes
with open(MODEL_PATH + 'labels.txt') as f:
    classes = f.read().split('\n')

# Define colors for masks
mask_colors = np.random.randint(125, 255, (len(classes), 3))


def draw_objects(image, boxes, masks, conf_threshold=0.5):
    """Extract the bounding box and mask for each detected object"""
    height = image.shape[0]
    width = image.shape[1]

    number_of_objects = boxes.shape[2]
    for i in range(number_of_objects):
        box = boxes[0, 0, i]
        conf = box[2]
        if conf > conf_threshold:
            # Extract the bounding box
            left = int(width * box[3])
            top = int(height * box[4])
            right = int(width * box[5])
            bottom = int(height * box[6])

            # Limit the bounding box values
            left = max(0, min(left, width - 1))
            top = max(0, min(top, height - 1))
            right = max(0, min(right, width - 1))
            bottom = max(0, min(bottom, height - 1))

            # Extract the mask for the object
            class_id = int(box[1])
            mask = masks[i][class_id]

            # Draw bounding box, colorize and show the mask on the image
            draw(image, mask, class_id, conf, left, top, right, bottom)


def draw(image, mask, class_id, conf, left, top, right, bottom,
         mask_threshold=0.3, color_per_class=True):
    """Draw a box, colorized mask, class and confidence on the image"""
    # Draw a bounding box
    cv.rectangle(image, (left, top), (right, bottom), (255, 178, 50), 3)

    # Display the label at the top of the bounding box
    label = f'{classes[class_id]}: {100*conf:.2f} %'
    label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5,
                                           1)
    top = max(top, label_size[1])
    cv.rectangle(image, (left, top - round(1.5 * label_size[1])),
                 (left + round(1.5 * label_size[0]), top + base_line),
                 (255, 255, 255), cv.FILLED)
    cv.putText(image, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75,
               (0, 0, 0), 1)

    # Resize the mask, threshold, color and apply it on the image
    mask = cv.resize(mask, (right - left + 1, bottom - top + 1))
    mask = (mask > mask_threshold)
    roi = image[top:bottom+1, left:right+1][mask]

    # Choose a color for the mask
    if color_per_class:
        color = mask_colors[class_id % len(mask_colors)]
    else:
        color_index = random.randint(0, len(mask_colors) - 1)
        color = mask_colors[color_index]

    # Overlay mask
    image[top:bottom+1, left:right+1][mask] = (
            [0.3 * color[0], 0.3 * color[1], 0.3 * color[2]] + 0.7 * roi
    ).astype(np.uint8)

    # Draw contours on the image
    # mask = mask.astype(np.uint8)
    # im2, contours, hierarchy = cv.findContours(mask, cv.RETR_TREE,
    #                                            cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(image[top:bottom+1, left:right+1], contours, -1, color,
    #                 3, cv.LINE_8, hierarchy, 100)


def get_response_data(image: str) -> dict:
    # Read image
    img = cv.imread(image)

    if img is not None:
        # Create a blob from an image
        blob = cv.dnn.blobFromImage(img, swapRB=True, crop=False)

        # Set the input to the network
        model.setInput(blob)

        # Run the forward pass to get output from the output layers
        boxes, masks = model.forward(['detection_out_final', 'detection_masks'])

        # Extract the bounding box and mask for each of the detected objects
        draw_objects(img, boxes, masks)

        # Put efficiency information.
        t, _ = model.getPerfProfile()
        label = 'Mask-RCNN : Inference time: %.2f ms' % (
                    t * 1000.0 / cv.getTickFrequency())
        cv.putText(img, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # Show image
        cv.imshow('Image', img)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    img_path = '/Users/admin/Desktop/Foto to move/2011.08 - Фестиваль_Братья_5.jpeg'
    get_response_data(img_path)

    # img = cv.imread(img_path)
    # img = cv.resize(img, (500, 500))
    # height, width, _ = img.shape
    # blank = np.zeros((height, width, 3), np.uint8)
    # cv.rectangle(blank, (-100, 100),
    #              (200, 600),
    #              (255, 0, 255), cv.FILLED)
    # # Show image
    # cv.imshow('Image', blank)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

# cap = cv.VideoCapture(img_path)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
#
# while True:
#     ret, image = cap.read()
#     if not ret:
#         print("Can't receive image (stream end?). Exiting ...")
#         break
#
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#
#     cv.imshow('image', gray)
#     if cv.waitKey(4000) == ord('q'):
#         break
#
# cap.release()
# cv.destroyAllWindows()

