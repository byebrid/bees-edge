import cv2
import json
import numpy as np
from typing import Union, Tuple, List, Dict
import string
from datetime import datetime as dt
from pathlib import Path

# profile = line_profiler.LineProfiler()

# from window import Window

def config(key, default=None):
    """Helper method to return value of `key` in `config.json`, else returns `default`"""
    with open("low_cost_mode/my_config.json", "r") as f:
        d = json.load(f)
    return d.get(key, default)


KEYS = {
    "space": 32,
    "left": 91, # Note this is actually "[", since left/right arrows are bugged in opencv with QT GUI (https://github.com/opencv/opencv/issues/20215)
    "right": 93, # This is "]"
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
        self.min_area = min_area
        self.reset_HSV() # Not actually resetting obviously

        self.video_src = video_src
        self.video_is_playing = True
        self.frame_changed = False
        
        # Image processing parameters
        self.movement_thresh = movement_thresh
        self.movement_pad = movement_pad
        self.color_thresh = color_thresh
        self.color_pad = color_pad
        self.dilate = dilate

        # This is lower bound (in ms) of delay opencv waits for keyentry.
        self.key_delay = 1

    def reset_HSV(self, *args):
        self.low_H = 179
        self.low_S = 255
        self.low_V = 255
        self.high_H = 0
        self.high_S = 0
        self.high_V = 0

    def get_video_cap(self):
        return cv2.VideoCapture(self.video_src)

    # @profile
    def display_video(self, show=True, write=False):

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
                    self.frame_changed = True
                if s < self.low_S:
                    print(f"Low Sat {self.low_S} -> {s}")
                    self.low_S = int(s)
                    self.frame_changed = True
                if v < self.low_V:
                    print(f"Low Val {self.low_V} -> {v}")
                    self.low_V = int(v)
                    self.frame_changed = True
                if h > self.high_H:
                    print(f"High Hue {self.high_H} -> {h}")
                    self.high_H = int(h)
                    self.frame_changed = True
                if s > self.high_S:
                    print(f"High Sat {self.high_S} -> {s}")
                    self.high_S = int(s)
                    self.frame_changed = True
                if v > self.high_V:
                    print(f"High Val {self.high_V} -> {v}")
                    self.high_V = int(v)
                    self.frame_changed = True

        def on_trackbar_movement_thresh(val):
            self.movement_thresh = val
            self.frame_changed = True

        def on_trackbar_frame(val):
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, val)
            self.frame_changed = True

        def on_trackbar_speed(val):
            self.key_delay = val
            self.frame_changed = True
    
        try:
            # Get video capture
            video_cap = self.get_video_cap()
            self.video_cap = video_cap

            if write:
                out_dir = Path("./out") / str(dt.now())
                Path.mkdir(out_dir)
                # Get some metadata for videos
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                fps = int(video_cap.get(cv2.CAP_PROP_FPS))
                width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # opencv only accepts string filepaths, not Path objects
                input_write_file = str(out_dir / "input.avi")
                diff_write_file = str(out_dir / "diff.avi")
                trail_write_file = str(out_dir / "trail.avi")

                input_video_writer = cv2.VideoWriter(input_write_file, fourcc=fourcc, fps=fps, frameSize=(width, height))
                diff_video_writer = cv2.VideoWriter(diff_write_file, fourcc=fourcc, fps=fps, frameSize=(width, height))
                trail_video_writer = cv2.VideoWriter(trail_write_file, fourcc=fourcc, fps=fps, frameSize=(width, height))
                # TODO: Figure out how to choose color at start?
                # color_video_writer = cv2.VideoWriter()
            
            # Get some metadata about video
            total_frames = self.total_frames

            # Define windows here for clarity
            if show:
                cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
                cv2.namedWindow("Difference", cv2.WINDOW_NORMAL)
                cv2.namedWindow("Color", cv2.WINDOW_NORMAL)
                cv2.namedWindow("Trail", cv2.WINDOW_NORMAL)

            if show:
                # Listen for mouse clicks on main video
                cv2.setMouseCallback("Input", on_mouse)
                # Add slider for threshold on motion detection
                cv2.createTrackbar("Threshold", "Difference", self.movement_thresh, 255, on_trackbar_movement_thresh)
                # Add slider for frame seek on main video
                cv2.createTrackbar("Frame", "Input", self.current_frame_index, self.total_frames-1, on_trackbar_frame)
                # Add slider for playback speed on main video
                cv2.createTrackbar("Delay", "Input", self.key_delay, 1000, on_trackbar_speed)
                cv2.setTrackbarMin("Delay", "Input", 1)
                # Add "slider" to act like button to reset colour thresholds in color window
                cv2.createButton("Reset color", self.reset_HSV)

            # Initialise previous frame which we'll use for motion detection/frame difference
            prev_frame = None
            # Trail frame is a cumulative frame of all movement detection frames.
            trail_frame = None
            # This is to help keep track of which pixels have already been set in trail frame. Doing this to try and get more bees
            # rather than flowers in final image!
            trail_updates = None

            # Play video, with all processing
            while True:
                if self.current_frame_index % 1000 == 0:
                    print(f"Frame {self.current_frame_index}/{self.total_frames}")

                if show:
                    # Check to see if user presses key
                    key = cv2.waitKey(self.key_delay)
                    key &= 0xFF
                    if key != KEYS["none"]:
                        keyname = KEYCODE_TO_NAME.get(key, "unknown")
                        print(f"Keypress: {keyname}  (code={key})")

                if self.video_is_playing or self.frame_changed:
                    if self.frame_changed and not self.video_is_playing:
                        self.current_frame_index -= 1
                    # Read next video frame, see if we've reached end
                    frame_grabbed, frame = video_cap.read()
                    if not frame_grabbed:
                        break
                    self.frame = frame # TODO: Clean this up (i.e. do we want self.frame or not?)
                    # Update frame slider with current frame
                    self.frame_changed = False
                
                # NOTE: Setting trackbar pos actually triggers an event on that
                # trackbar. My listener does some expensive updates for that event,
                # so I've commented out this live updating of trackbar.
                # if show:
                #     cv2.setTrackbarPos("Frame", "Input", self.current_frame_index)
                
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

                # Update trail frame too
                if trail_frame is None:
                    trail_frame = diff_frame.copy()
                    trail_updates = np.zeros(trail_frame.shape, dtype=bool)
                    trail_updates[diff_mask] = True
                else:
                    # Double-check that none of these pixels have been updated before
                    if not np.any(trail_updates, where=diff_mask):
                        trail_frame[diff_mask] = frame[diff_mask]
                        trail_updates[diff_mask] = True
                
                # Copy normal frame so we can add frame number without messing up original frame
                if show:
                    frame_copy = frame.copy()
                    i = self.current_frame_index
                    cv2.putText(frame_copy, f"Frame {i}/{total_frames}", (50, 100), self.FONT, 2, (0, 0, 255), 3)
                    frame_copy

                    # Show frames of interest
                    cv2.imshow("Input", frame_copy)
                    cv2.imshow("Difference", diff_frame)
                    cv2.imshow("Color", color_frame)
                    cv2.imshow("Trail", trail_frame)

                # Update previous frame now only if frame
                if self.video_is_playing:
                    prev_frame = frame

                if write:
                    # input_video_writer.write(frame)
                    diff_video_writer.write(diff_frame)
                    trail_video_writer.write(trail_frame)

                # Respond to key press
                if show:
                    if key == ord("q"):
                        break
                    elif key == KEYS["space"]:
                        self.on_play_pause()
                    elif key == KEYS["left"]:
                        self.seek_relative(-1)
                    elif key == KEYS["right"]:
                        self.seek_relative(+1)

            # # At very end of loop, let's save that trail frame to an image
            # if write:
            #     trail_write_file = str(out_dir / "trail.jpg")
            #     cv2.imwrite(trail_write_file, trail_frame)

        except Exception as e:
            raise e
        finally:
            print("Releasing video capture and closing all opencv windows")
            video_cap.release()
            cv2.destroyAllWindows()
            if write:
                print("Releasing video writers")
                input_video_writer.release()
                diff_video_writer.release()
                trail_video_writer.release()

    @property
    def current_frame_index(self) -> int:
        if self.video_cap is not None:
            return int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        return -1

    @current_frame_index.setter
    def current_frame_index(self, val):
        if self.video_cap is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, val)
            self.frame_changed = True
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
    
    def seek_relative(self, frame_delta:int,*args, **kwargs):
        """
        Increment frame index by `frame_delta`. I.e., if `frame_delta` is positive,
        this seeks right; if negative, it seeks left.
        """
        i = self.current_frame_index
        new_i = i + frame_delta
        if new_i >= 0 and new_i < self.total_frames:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, new_i)
            self.frame_changed = True
        else:
            print(f"Can't change frame from {i} to {new_i}!")

    def set_frame(self, i):
        self.current_frame_index = i

    # @profile
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

        mask = np.zeros(frame.shape, bool) # Array of all False
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

            mask[y0:y1, x0:x1] = True

        return mask



D = Display(video_src=config("INPUT_FILEPATH"), movement_pad=5, movement_thresh=80)
D.display_video(write=True, show=True)