# y_util.py

import pickle
from skimage.transform import resize
import numpy as np
import cv2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

EMPTY = True
NOT_EMPTY = False

# Load model path from .env
MODEL_PATH = os.getenv("MODEL_PATH")
MODEL = pickle.load(open(MODEL_PATH, "rb"))


def empty_or_not(spot_bgr):
    """
    Predict whether a parking spot is empty or not using the trained model.

    :param spot_bgr: The image of the parking spot in BGR format.
    :return: True if the spot is empty, False otherwise.
    """
    flat_data = []

    img_resized = resize(spot_bgr, (15, 15, 3))  # Resize the image to match the model's input size
    flat_data.append(img_resized.flatten())  # Flatten the image
    flat_data = np.array(flat_data)  # Convert to NumPy array

    y_output = MODEL.predict(flat_data)  # Make prediction

    return EMPTY if y_output == 0 else NOT_EMPTY


def find_connected_components(mask):
    """
    Find connected components in a given mask image.

    :param mask: Binary (black and white) mask image where components need to be detected.
    :return: Labels and properties of the connected components.
    """
    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    return connected_components


def get_parking_spots_bboxes(connected_components):
    """
    Identify parking spots using connected component labels and properties.

    :param connected_components: Labels and properties of the connected components.
    :return: A list of bounding box coordinates for each parking spot (x1, y1, w, h).
    """
    (totalLabels, label_ids, values, centroid) = connected_components
    slots = []
    coef = 1  # Scaling coefficient (if needed for resizing)

    for i in range(1, totalLabels):
        # Extract coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)
        slots.append([x1, y1, w, h])

    return slots
