# y-main.py

import os
import cv2
from dotenv import load_dotenv
from y_util import empty_or_not, get_parking_spots_bboxes, find_connected_components

# Load environment variables
load_dotenv()

# Get file paths from .env
mask_path = os.getenv("MASK_PATH")

# Read mask image
mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask_height, mask_width = mask_img.shape

# Camera connection
cap = cv2.VideoCapture(0)  # The camera index may vary depending on the connected camera

# Capture an initial frame to determine camera resolution
ret, frame = cap.read()

if not ret:
    print("Failed to capture video!")
    exit()

# Get screen size for dynamic scaling
screen_width = 1280  # Default width (change if needed)
screen_height = 720  # Default height (change if needed)

# Resize frame to fit the screen while maintaining aspect ratio
aspect_ratio = frame.shape[1] / frame.shape[0]
if screen_width / screen_height > aspect_ratio:
    frame_height = screen_height
    frame_width = int(frame_height * aspect_ratio)
else:
    frame_width = screen_width
    frame_height = int(frame_width / aspect_ratio)

frame = cv2.resize(frame, (frame_width, frame_height))

# Resize mask image to match the new frame size
mask_img = cv2.resize(mask_img, (frame_width, frame_height))

# Find connected components in the mask
connected_components = find_connected_components(mask_img)
spots = get_parking_spots_bboxes(connected_components)
print(spots)

ret = True
while ret:
    ret, frame = cap.read()

    if not ret:
        print("Camera not detected")
        break

    # Resize the frame dynamically based on screen resolution
    frame = cv2.resize(frame, (frame_width, frame_height))

    for spot_index, spot in enumerate(spots):
        x1, y1, w, h = spot

        # Crop the detected parking spot
        spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

        # Determine if the spot is empty or occupied
        spot_status = empty_or_not(spot_crop)

        if spot_status:
            color = (0, 255, 0)  # Green (occupied)
        else:
            color = (0, 0, 255)  # Red (empty)

        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    # Resize window to fit any screen while maintaining proportions
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', screen_width, screen_height)

    cv2.imshow('Frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
