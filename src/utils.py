import numpy as np
import cv2


def refine_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    ret_mask: bool,
    thresh: int,
    pad: int,
    color=cv2.COLOR_BGR2GRAY,
    dilate: int = None,
    min_area: int = 0,
):
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

    Parameters
    ----------
    frame: numpy.ndarray
        The original input frame of the data (i.e. not processed!)
    mask: numpy.ndarray
        The pre-processed frame which we'll treat as the rough mask to smoothen out.
    ret_mask: bool
        Whether to return a mask or not. If False, then we return a masked copy
        of the original frame. If True, we return a True/False mask array. If you
        only want a masked version of the input frame, and don't need the mask
        for anything else, setting this to False will be faster!
    TODO: rest of parameters!
    """
    if color is not None:
        gray = cv2.cvtColor(mask, color)
    else:  # sometimes we use an already-gray input
        gray = mask.copy()
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

    if ret_mask:
        ret = np.zeros(mask.shape, dtype=bool)  # Array of all False

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

            # TODO: Test how much faster it is if we literally copy-paste this for
            # loop into the single if-else statement above for mask vs frame. We're
            # doing an if statement for every contour, which isn't terrible but could
            # add up!
            ret[y0:y1, x0:x1] = True
    else:
        ret = np.zeros(frame.shape, dtype=np.uint8)

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

            # TODO: Test how much faster it is if we literally copy-paste this for
            # loop into the single if-else statement above for mask vs frame. We're
            # doing an if statement for every contour, which isn't terrible but could
            # add up!
            ret[y0:y1, x0:x1] = frame[y0:y1, x0:x1]

    return ret


def detect_motion(
    frame: np.ndarray, prev_frame: np.ndarray, thresh: int = 40, kernel_size: int = 63
) -> np.ndarray:
    """
    Returns a mask for the given `frame` that tries to include only the "moving"
    objects in the image.

    :param frame: Current frame of video
    :param prev_frame: Previous frame of video
    :param thresh: Minimum threshold to define what is and is not "movement" , defaults to 40
    :param kernel_size: Width/height of kernel used to dilate the mask (provides a more filled-in image of each moving object), defaults to 63
    :return: A mask array (i.e. boolean) that can be used to index the given `frame` to only pull out the "moving" parts of the frame
    """
    ###### RESIZING
    # orig_shape = frame.shape
    # fx = 0.25
    # fy = 0.25
    # frame = cv2.resize(frame, dsize=None, fx=fx, fy=fy)
    # prev_frame = cv2.resize(prev_frame, dsize=None, fx=fx, fy=fy)

    # Compute pixel difference between consecutive frames (note this still has 3 channels)
    diff = cv2.absdiff(frame, prev_frame)
    # Convert to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Cut off pixels that did not have "enough" movement. This is now a 2D array
    # of just 1s and 0s
    _, mask = cv2.threshold(
        src=gray, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY
    )
    
    ###### RESIZING
    # Since we now have a very choppy/pixelly mask, we want to blur it out a bit.
    # This mimics finding the bounding boxes of bees (or whatever objects are
    # moving), but does so in a much cheaper/faster way!
    # kernel_size = int(kernel_size * fx) # Scale the kernel size to keep things consistent 
    # mask = cv2.dilate(threshed, kernel=np.ones((kernel_size, kernel_size)))
    # # Up-res the final mask (note opencv expects opposite order of dimensions because of course it does)
    # mask = cv2.resize(mask, dsize=(orig_shape[1],orig_shape[0]))

    # Convert to boolean so we can actually use it as a mask now
    mask = mask.astype(bool)

    return mask


def detect_motion_old_way(frame: np.ndarray, prev_frame: np.ndarray, thresh: int, pad: int) -> np.ndarray:
    diff = cv2.absdiff(frame, prev_frame)
    return refine_mask(frame=frame, mask=diff, ret_mask=True, thresh=thresh, pad=pad)