import cv2
import numpy as np

def extract_color_vector(image_path, bins=8):
    """
    Extracts a simple color representation from an image.

    Uses:
    - HSV color space
    - Histogram over Hue channel

    This provides a lightweight color signal used for
    soft re-ranking (not strict filtering).
    """
    # Load image using OpenCV
    img = cv2.imread(image_path)

    # Convert from BGR (OpenCV default) to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Compute histogram over Hue channel
    hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist = hist.flatten()

    # Normalize histogram
    return hist / np.sum(hist)
