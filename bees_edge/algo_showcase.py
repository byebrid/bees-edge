"""
This is intended to be a temporary file that visualises each step of the 
motion-detection algorithm.

Note I'm not even trying to reuse my existing code properly since this 
visualisation is a very different use-case, so just copy-pasting as needed.
"""
import cv2
import numpy as np

from pathlib import Path

# Filepath of input video
INPUT_FP = Path("/home/lex/Development/bees-edge/data/micro_cam_8_N_video_20210310_133844.h264.avi")
# Name of window to show processed video
WINDOW_NAME = "Motion detection showcase"
# Factor by which to downscale video
DOWNSCALE_FACTOR = 0.5
# Threshold to apply to pixel intensity differences
MOVEMENT_THRESHOLD = 40
# Kernel size to dilate motion mask (note this is in terms of *original* 
# resolution, it will be updated automatically based on downscale factor)
DILATION_KERNEL_SIZE = 128
_DOWNSCALED_DILATION_KERNEL_SIZE = int(128 * DOWNSCALE_FACTOR)

# The actual dilation kernel
DILATION_KERNEL = np.ones((_DOWNSCALED_DILATION_KERNEL_SIZE, _DOWNSCALED_DILATION_KERNEL_SIZE))

class Quit(Exception):
    pass


def wait_for_keypress(key: str = "c", interval_ms: int = 25):
    """Do nothing until certain keypress received by opencv window."""
    if len(key) != 1:
        raise ValueError("Keypress should only be a single character")
    
    print(f"Press {key} to continue (or 'q' to quit)")

    while True:
        keypress = cv2.waitKey(interval_ms) & 0xFF
        if keypress == ord(key):
            return
        elif keypress == ord("q"):
            raise Quit("User pressed 'q' to exit")


# Get handle to read video
video_cap = cv2.VideoCapture(filename=str(INPUT_FP))
# Main window to show each step of the flow
window = cv2.namedWindow("Motion detection showcase", cv2.WINDOW_NORMAL)
# Store previous frame (used for computing intensity difference between frames)
prev_gray = None
    

try:
    while True:
        # STEP 1: Show input frame
        grabbed, frame = video_cap.read()
        if not grabbed or frame is None:
            break

        print("STEP 1: Input frame")
        cv2.imshow(WINDOW_NAME, frame)
        wait_for_keypress()

        # STEP 2: Downscale
        DOWNSCALE_FACTOR = 0.5
        downscaled_frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
        
        print("STEP 2: Downscale")
        cv2.imshow(WINDOW_NAME, downscaled_frame)
        wait_for_keypress()

        # STEP 3: Compute change in intensity of each pixel compared to previous frame
        if prev_gray is None:
            prev_gray = cv2.cvtColor(downscaled_frame, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.cvtColor(downscaled_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        
        print("STEP 3: Convert to grayscale and compute change in pixel intensities")
        cv2.imshow(WINDOW_NAME, diff)
        wait_for_keypress()

        # STEP 4: Apply threshold to only select "large movements"
        # Cut off pixels that did not have "enough" movement. This is now a 2D array
        # of just 1s and 0s
        _, threshed_diff = cv2.threshold(
            src=diff, thresh=MOVEMENT_THRESHOLD, maxval=255, type=cv2.THRESH_BINARY
        )

        print("STEP 4: Apply threshold to only select 'large movements'")
        cv2.imshow(WINDOW_NAME, threshed_diff)
        wait_for_keypress()

        # STEP 5: Dilate that motion mask
        mask = cv2.dilate(threshed_diff, kernel=DILATION_KERNEL)

        print("STEP 5: Dilate that motion mask")
        cv2.imshow(WINDOW_NAME, mask)
        wait_for_keypress()

        # STEP 6: Upscale that motion mask back to original resolution
        mask = cv2.resize(mask, dsize=(frame.shape[1], frame.shape[0]))

        print("STEP 6: Upscale the motion mask back to full-resolution")
        cv2.imshow(WINDOW_NAME, mask)
        wait_for_keypress()

        # STEP 7: Apply that mask to original frame, giving us the final output
        mask = mask.astype(bool)

        # Save downscaled frame for use in next iteration
        prev_gray = gray
        # Note that we don't save the thresholded diff here. Otherwise, movement
        # could only persist across single frames at most, which is no good!
        prev_diff = diff

        # Return the final frame with only regions of "movement" included, everything
        # else blacked out
        motion_frame = np.zeros(shape=frame.shape, dtype=np.uint8)
        motion_frame[mask] = frame[mask]

        print("STEP 7: Apply mask back to original frame")
        cv2.imshow(WINDOW_NAME, motion_frame)
        wait_for_keypress()

except Quit:
    pass
except (Exception, KeyboardInterrupt) as e:
    print("Something went wrong. Exiting...")
    print(e)
finally:
    # Clean up resources
    video_cap.release()
    cv2.destroyAllWindows()
