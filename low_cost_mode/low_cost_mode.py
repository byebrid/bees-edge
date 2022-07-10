"""
A lot taken from https://pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/.

This file is intended for experimenting with simple methods to coarsely identify possible bees and flowers to be
used on an edge device.
"""
# %%
# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import json
from typing import Tuple
from pathlib import Path

def config(key, default=None):
    """Helper method to return value of `key` in `config.json`, else returns `default`"""
    with open("my_config.json", "r") as f:
        d = json.load(f)
    return d.get(key, default)

# # %%
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", help="path to the video file")
# ap.add_argument("-a", "--min-area", type=int, default=0, help="minimum area size")
# args = vars(ap.parse_args())

# %%
# Read video
vs = cv2.VideoCapture(config("INPUT_FILEPATH"))

# %%
# Store previous frame to compare with current frame to do backgroumd subtraction
prev_frame = None

# %%
# Figure out dimensions/fps of video
width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)) # NOTE: Could change this if we're happy to downsample, though feels unnecessary if we're removing frames/pixels anyway!
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = vs.get(cv2.CAP_PROP_FPS)

# %%
# Check desired encoding for output
fourcc = cv2.VideoWriter_fourcc(*config("FOURCC", ["XVID"])) 

# %%
# Figure out output video filepath
output_dir = Path(config("OUTPUT_DIRECTORY"))
output_filepath = output_dir / str(datetime.datetime.now())
output_filepath = output_filepath.with_suffix(config("OUTPUT_EXT", ".avi"))
# Convert to string for opencv
output_filepath = str(output_filepath)

# %%
# Make the actual video object
video = cv2.VideoWriter(filename=output_filepath, fourcc=fourcc, fps=frame_rate, frameSize=(width, height))

# %%
def get_flowers_frame(
        frame: np.ndarray,
        low_colour_bound: Tuple[int] = (130, 0, 0),
        upp_colour_bound: Tuple[int] = (155, 255, 240),
        box_padding: int = 15,
        min_area: int = 10,
    ) -> np.ndarray:
    """
    Returns a copy of the given frame with only the pixels of or near flowers 
    included. All other pixels are blacked out. When we say "flowers", we really 
    just mean any pixels with HSV colours between the given bounds.

    Parameters
    ----------
    frame: numpy array representing frame of video. Expected to be BGR colour 
        scheme.
    low_colour_bound: tuple of 3 integers representing the lower bound of HSV 
        values to include in the final image. Note that opencv's Hue values are
        in range [0, 179].
    upp_colour_bound: same as above, but upper bound.
    box_padding: Padding to be used around bounding boxes to make sure we 
        include areas just next to flowers too (since bees can pollinate without 
        being exactly on top of flower).
    min_area: Minimum area formed by a contour for it to be considered 
        significant. Higher values help to drop irrelevant one-pixel changes that
        don't represent a bee or flower, but too high and you'll lose important
        data.
    dilation_kernel: Width/height of kernel used for dilating, which is what 
        helps us join coloured regions together before we find their contours.
        The bigger this is, the "blurrier" your thresholding will be, and the 
        larger the area of your contours (roughly).
    """
    # Convert to HSV colour mode for easier colour thresholding
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, low_colour_bound, upp_colour_bound)
    colour_mask = mask>0
    flower_hsv = np.zeros_like(frame_hsv, np.uint8)
    flower_hsv[colour_mask] = frame_hsv[colour_mask]

    # Convert to BGR since that's what opencv expects when using imshow()
    flower_frame = cv2.cvtColor(flower_hsv, cv2.COLOR_HSV2BGR)

    # Now try to binarise this so we can find contours
    flower_frame_gray = cv2.cvtColor(flower_frame, cv2.COLOR_BGR2GRAY)

    # Dilate to join nearby regions for better contours
    kernel = np.ones((3, 3))
    flower_frame_gray_dilated = cv2.dilate(flower_frame_gray, kernel, iterations=1)

    # Threshold to binarise the image, and find contours
    _, flower_frame_bin = cv2.threshold(flower_frame_gray_dilated, 1, 255, 0) # Literally any non-black pixel
    contours, hierarchy = cv2.findContours(flower_frame_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # This is copy of original frame, only with flower/adjacent pixels included
    flower_frame_masked_rect = np.zeros(flower_frame.shape, np.uint8)

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        # Instead of drawing the tight contours, draw their rectangular bounding boxes
        box = cv2.boundingRect(contour)
        
        # Get coordinates of corners of bounding box, including padding
        x, y, w, h = box
        x0 = x - box_padding
        y0 = y - box_padding
        x1 = x + w + box_padding
        y1 = y + h + box_padding

        flower_frame_masked_rect[y0:y1, x0:x1] = frame[y0:y1, x0:x1]

    return {
        "flower_pixels": flower_frame,
        "binarised": flower_frame_bin,
        "rect_mask": flower_frame_masked_rect
    }

# %%
vs = cv2.VideoCapture(config("INPUT_FILEPATH"))
frame_grabbed, frame = vs.read()
frame_dict = get_flowers_frame(frame, box_padding=10, low_colour_bound=(30, 0, 0), upp_colour_bound=(35, 255, 255))

while True:
    # cv2.drawContours(flower_frame, contours, -1, (0, 255, 0))
    # cv2.imshow("Masked video", flower_frame)
    # cv2.imshow("Binarised mask", flower_frame_bin)
    cv2.imshow("Rect. masked video", frame_dict["rect_mask"])
    cv2.imshow("Raw masked video", frame_dict["flower_pixels"])
    cv2.imshow("Binarised", frame_dict["binarised"])

    # Check to see if user presses key
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
vs.release()
cv2.destroyAllWindows()

# %%
# Read video
vs = cv2.VideoCapture(config("INPUT_FILEPATH"))

while True:
    # grab the current frame
    frame_grabbed, frame = vs.read()
    
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not frame_grabbed:
        break
    
    # Need previous frame to compare to, this initialises it
    if prev_frame is None:
        prev_frame = frame
        continue # Ignore first frame since nothing to compare it to
        
    # # compute the absolute difference between the current frame and
    # # first frame
    # frame_delta = cv2.absdiff(prev_frame, frame)
    # thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    # # dilate the thresholded image to fill in holes, then find contours
    # # on thresholded image

    # # Finally, write to output video
    # video.write(masked_frame)

    # Check to see if user presses key
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
vs.release()
# vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
video.release()
# %%
