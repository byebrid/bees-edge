import numpy as np
import cv2

def get_mask(frame: np.ndarray, thresh: int, pad: int, color=cv2.COLOR_BGR2GRAY, dilate:int=None, min_area:int=0):
    """
    Convert a 'soft mask' (e.g. difference between two frames, or all pixels
    within some range of HSV values) into a masked copy of frame using
    rectangular bounding boxes. Specifically, we do the following:

        1) Convert to grayscale
        2) Threshold
        3) Dilate
        4) Find contours
        5) Find bounding boxes of those contours
        6) Return masked copy of frame using those bounding boxes
    """
    if color is not None:
        gray = cv2.cvtColor(frame, color)
    else:  # sometimes we use an already-gray input
        gray = frame.copy()
    _, threshed = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    # Only dilate if some kernel size was given
    if dilate is not None:
        dilate_kernel = np.ones((dilate, dilate))
        dilated = cv2.dilate(threshed, dilate_kernel)
    else:
        dilated = threshed

    contours, hierarchy = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    mask = np.zeros(frame.shape, bool)  # Array of all False
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        # Instead of using the exact contours, draw their rectangular bounding boxes
        x0, y0, w, h = cv2.boundingRect(contour)

        x1 = x0 + w
        y1 = y0 + h

        x0 -= pad
        y0 -= pad
        x1 += pad
        y1 += pad

        mask[y0:y1, x0:x1] = True

    return mask
