import cv2
import numpy as np


def capture_rgb_screen(sct, bbox: dict) -> np.ndarray:
    """
    Capture a screen region and return it as a numpy array.
    
    Args:
        sct: mss object
        bbox: dictionary containing the coordinates and size of the box to capture
    
    Returns:
        numpy.ndarray: the screen region as a numpy array
    """
    sct_img = sct.grab(bbox)
    img_rgba = np.array(sct_img)
    return cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)