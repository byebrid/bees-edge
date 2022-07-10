import cv2
import json
import numpy as np
from typing import Union, Tuple, List, Dict
import string

# from window import Window

def config(key, default=None):
    """Helper method to return value of `key` in `config.json`, else returns `default`"""
    with open("low_cost_mode/my_config.json", "r") as f:
        d = json.load(f)
    return d.get(key, default)


KEYS = {
    "space": 32,
    "left": 81,
    "up": 82,
    "right": 83,
    "down": 84,
    "esc": 27,
    "none": 255,
}
KEYS.update({l: ord(l) for l in string.ascii_lowercase})
KEYCODE_TO_NAME = {v: k for k, v in KEYS.items()}


class Display:
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, *, video_src, min_area:int=10, dilate:int=3, movement_thresh=40, color_thresh=30, movement_pad=20, color_pad=5):
        # TODO: Noticing that I need a min_area, dilate, thresh, and padding for
        # both motion detection AND colour masking. Need to think of nice way of
        # doing this
        self.HSVs = {
            "low": {
                "H": 30,
                "S": 0,
                "V": 0
            },
            "upp": {
                "H": 35,
                "S": 255,
                "V": 255
            }
        }
        self.min_area = min_area

        self.video_src = video_src
        self.video_is_playing = True
        
        # Image processing parameters
        self.movement_thresh = movement_thresh
        self.movement_pad = movement_pad
        self.color_thresh = color_thresh
        self.color_pad = color_pad
        self.dilate = dilate

    def get_video_cap(self):
        return cv2.VideoCapture(config("INPUT_FILEPATH"))

    # def on_change_hue(self, event, low, key):
    #     # Key should be one of 'H', 'S', 'V'
    #     if low:
    #         d = self.HSVs["low"]
    #     else:
    #         d = self.HSVs["upp"]
    #     d[key] = event
    #     self.change_flag = True

    # def get_HSV(self, low=True):
    #     if low:
    #         d = self.HSVs["low"]
    #     else:
    #         d = self.HSVs["upp"]
    #     return (d["H"], d["S"], d["V"])

    # @profile
    def display_video(self):
        self.low_H = 179
        self.low_S = 255
        self.low_V = 255
        self.high_H = 0
        self.high_S = 0
        self.high_V = 0

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                h, s, v = hsv[y, x]

                # Could use min/max to do these, but I'm using if statements
                # so it's easier to spot when a value has been updated (so user
                # can tell when their clicks aren't actually doing anything!)
                # TODO: Clean this up, maybe similar to dict apporach I've already used
                # before for HSV
                if h < self.low_H:
                    print(f"Low Hue {self.low_H} -> {h}")
                    self.low_H = int(h)
                if s < self.low_S:
                    print(f"Low Sat {self.low_S} -> {s}")
                    self.low_S = int(s)
                if v < self.low_V:
                    print(f"Low Val {self.low_V} -> {v}")
                    self.low_V = int(v)
                if h > self.high_H:
                    print(f"High Hue {self.high_H} -> {h}")
                    self.high_H = int(h)
                if s > self.high_S:
                    print(f"High Sat {self.high_S} -> {s}")
                    self.high_S = int(s)
                if v > self.high_V:
                    print(f"High Val {self.high_V} -> {v}")
                    self.high_V = int(v)

        def on_trackbar_movement_thresh(val):
            self.movement_thresh = val

        def on_trackbar_frame(val):
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, val)

        try:
            # Get video capture
            video_cap = self.get_video_cap()
            self.video_cap = video_cap
            
            # Get some metadata about video
            total_frames = self.total_frames

            # Define windows here for clarity
            cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Difference", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Color", cv2.WINDOW_NORMAL)

            # Listen for mouse clicks on main video
            cv2.setMouseCallback("Input", on_mouse)
            # Add slider for threshold on motion detection
            cv2.createTrackbar("Threshold", "Difference", self.movement_thresh, 255, on_trackbar_movement_thresh)
            # Add slider for frame seek on main video
            cv2.createTrackbar("Frame", "Input", 0, self.total_frames, on_trackbar_frame)

            # Initialise previous frame which we'll use for motion detection/frame difference
            prev_frame = None

            # Play video, with all processing
            while True:
                # Check to see if user presses key
                key = cv2.waitKey(1)
                key &= 0xFF
                if key != KEYS["none"]:
                    keyname = KEYCODE_TO_NAME.get(key, "unknown")
                    print(f"Keypress: {keyname}  (code={key})")

                # Respond to key press
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, -1)
                    continue
                elif key == KEYS["space"]:
                    self.on_play_pause()
                    continue

                # If video playing, fetch **next** frame
                if self.video_is_playing:
                    # Read next video frame, see if we've reached end
                    frame_grabbed, frame = video_cap.read()
                    if not frame_grabbed:
                        break
                    self.frame = frame # TODO: Clean this up (i.e. do we want self.frame or not?)
                    cv2.setTrackbarPos("Frame", "Input", self.current_frame_index)
                else:
                    continue
                
                if prev_frame is None:
                    prev_frame = frame

                # Get mask based on movement
                diff = cv2.absdiff(frame, prev_frame)
                diff_mask = self.soft_mask_to_bounding_boxes(diff, thresh=self.movement_thresh, pad=self.movement_pad)

                # Get mask based on color
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv_low = (self.low_H, self.low_S, self.low_V)
                hsv_high = (self.high_H, self.high_S, self.high_V)
                inrange = cv2.inRange(hsv, hsv_low, hsv_high)
                color_mask = self.soft_mask_to_bounding_boxes(inrange, thresh=self.color_thresh, pad=self.color_pad, color=None)

                # Convert masks to actual frames containing masked pixel data
                diff_frame = np.zeros(frame.shape, np.uint8)
                diff_frame[diff_mask] = frame[diff_mask]
                color_frame = np.zeros(frame.shape, np.uint8)
                color_frame[color_mask] = frame[color_mask]
                
                # Copy normal frame so we can add frame number without messing up original frame
                frame_copy = frame.copy()
                i = int(video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                cv2.putText(frame_copy, f"Frame {i+1}/{total_frames}", (50, 100), self.FONT, 2, (0, 0, 255), 3)

                # Show frames of interest
                cv2.imshow("Input", frame_copy)
                cv2.imshow("Difference", diff_frame)
                cv2.imshow("Color", color_frame)

                # Update previous frame now
                prev_frame = frame

        except Exception as e:
            raise e
        finally:
            video_cap.release()
            cv2.destroyAllWindows()

    @property
    def current_frame_index(self) -> int:
        if self.video_cap is not None:
            return int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        return -1

    @current_frame_index.setter
    def current_frame_index(self, val):
        if self.video_cap is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, val)
        else:
            raise ValueError("Can't set new frame index because `self.video_cap` is undefined!")

    @property
    def total_frames(self) -> int:
        if self.video_cap is not None:
            return int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return -1

    def on_play_pause(self, *args, **kwargs):
        # Just toggle play/pause
        self.video_is_playing = not self.video_is_playing
    
    def rewind(self, *args, **kwargs):
        i = self.current_frame_index
        if i > 0:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, i - 1)
            self.frame_changed = True
        else:
            print(f"Can't rewind! Current frame number = {i}")

    def fast_forward(self, *args, **kwargs):
        i = self.current_frame_index
        if i < self.total_frames - 1:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, i + 1)
            self.frame_changed = True
        else:
            print(f"Can't fast-forward! Current frame number = {i} (total frames = {self.total_frames})")

    def set_frame(self, i):
        self.current_frame_index = i

    def set_thresh(self, thresh):
        self.thresh = thresh

    def soft_mask_to_bounding_boxes(self, frame:np.ndarray, thresh:int, pad: int, color=cv2.COLOR_BGR2GRAY):
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
        dilate_kernel = np.ones((self.dilate, self.dilate))
        
        if color is not None:
            gray = cv2.cvtColor(frame, color)
        else: # sometimes we use an already-gray input
            gray = frame.copy()
        _, threshed = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        
        dilated = cv2.dilate(threshed, dilate_kernel)
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros(frame.shape, np.uint8)
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            # Instead of using the exact contours, draw their rectangular bounding boxes
            x0, y0, w, h = cv2.boundingRect(contour)
            
            x1 = x0 + w
            y1 = y0 + h

            x0 -= pad
            y0 -= pad
            x1 += pad
            y1 += pad

            # mask[y0:y1, x0:x1] = frame[y0:y1, x0:x1]  # Note numpy arrays are rows, then columns, opposite order to opencv!
            mask[y0:y1, x0:x1] = 1

        return mask > 0



D = Display(video_src=config("INPUT_FILEPATH"))
# D.display()
D.display_video()