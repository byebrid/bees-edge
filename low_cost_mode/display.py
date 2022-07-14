import cv2
import json
import numpy as np
from typing import Union, Tuple, List, Dict
import string
from datetime import datetime as dt
from pathlib import Path
from queue import Queue
from threading import Thread
from tqdm import tqdm
from functools import lru_cache

# Just to deal with @profile decorator from line_profiler. It's a weird system but whatever
if type(__builtins__) is not dict or 'profile' not in __builtins__: profile=lambda f:f

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
# Add lowercase letters to dictionary as well so we better recognise user inputs
KEYS.update({l: ord(l) for l in string.ascii_lowercase})
# Keep reverse mapping to for when we know keycode, but want to know human-readable name
KEYCODE_TO_NAME = {v: k for k, v in KEYS.items()}


class ThreadedVideo:
    """
    Use this to perform read operations on a different thread to eliminate blocking
    times where possible.

    Stolen from https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/.
    """
    def __init__(self, path, queue_size=512):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False

        self.Q = Queue(maxsize=queue_size)
        self.t = None # To keep track of this Thread

    def start(self):
        # start a thread to read frames from the file video stream
        self.t = Thread(target=self.update)
        # t.daemon = True
        self.t.start()
        return self

    def get(self, key):
        return self.stream.get(key)

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                self.stream.release()
                return
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                grabbed, frame = self.stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
                # add the frame to the queue
                self.Q.put(frame)

    def read(self):
        # return next frame in the queue
        return True, self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def release(self):
        # indicate that the thread should be stopped
        self.stopped = True


class ThreadedVideoWriter:
    def __init__(self, *args, queue_size=64, flush_thresh=32, **kwargs):
        """

        Parameters
        ----------
        queue_size: int
            Maximum number of frames we are allowed to store in this writer's 
            queue.
        flush_thresh: int
            The number of frames we will allow in the queue before trying to
            write to a file. This should be less than `queue_size`, 
        """
        # Just validate parameters
        if flush_thresh >= queue_size:
            raise ValueError(f"flush_thresh ({flush_thresh}) should be considerable less than queue_size {queue_size}!")

        self.Q = Queue(maxsize=queue_size)
        self.flush_thresh = flush_thresh
        self.t = None # to keep track of current writing Thread
        self.video = cv2.VideoWriter(*args, **kwargs)
        self.dumping = False # to keep track of whether we are currently writing to file or not

    def release(self):
        """
        Flushes final frames in queue to file, making sure to wait for previous
        thread. 

        This is blocking, unlike the normal `write()`. 
        """
        # Dump remaining frames before closing thread
        self.start_thread(wait=True, block=True)

    def write(self, frame):
        """
        Meant to mimic `cv2.VideoWriter.write(image)`. This actually only adds
        the given `frame` to a queue.

        If the queue size exceeds `flush_thresh`, then this will trigger a flush
        of all frames in the queue into the file. See `start_thread()` for more
        details.
        """
        self.Q.put(frame)
        if self.Q.qsize() >= self.flush_thresh and not self.dumping:
            self.start_thread()

    def dump(self):
        """
        Writes all frames in queue to output file. Note if new frames are added
        to queue in meantime, these will also be written, meaning we typically
        write a few more than `flush_thresh` frames when we call this!
        """
        self.dumping = True

        while not self.Q.empty():
            frame = self.Q.get()
            self.video.write(frame)

        self.dumping = False
        return

    def thread_alive(self):
        return self.t and self.t.is_alive()

    def start_thread(self, wait=False, block=False):
        """
        This kicks off a new thread which starts writing all frames in the queue
        to the output file.

        Parameters
        ----------
        wait: bool
            Set this to True if you are expecting there to be a previous thread 
            still running. This might happen if you have finished going through
            a video, but still need to flush a few more frames right after.
            If False, then will raise a ValueError if a previous thread is already
            running.
        block: bool
            Whether to make the new thread blocking or not. Useful if you need to
            ensure this thread finishes before the main thread continues (i.e. at
            the end of execution)/
        """
        if self.dumping:
            if wait: # let old thread block main thread so it can finish before starting our next one.
                self.t.join()
            else:
                raise ValueError("Tried to dump but previous thread was still running!")
        self.t = Thread(target=self.dump)
        self.t.start()

        if block:
            self.t.join()


class Display:
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, *, video_src: str, out_dir: str, fourcc: int = None, min_area:int=10, dilate:int=3, movement_thresh:int=40, color_thresh:int=30, movement_pad:int=20, color_pad:int=5, downsample:float=1.0):
        # TODO: Noticing that I need a min_area, dilate, thresh, and padding for
        # both motion detection AND colour masking. Need to think of nice way of
        # doing this
        self.reset_HSV() # Not actually resetting obviously

        # Filepaths
        self.video_src = video_src
        self.out_dir = Path(out_dir)

        # Video encoding stuff
        if fourcc is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.fourcc = fourcc
        
        # Image processing parameters
        self.min_area = min_area
        self.movement_thresh = movement_thresh
        self.movement_pad = movement_pad
        self.color_thresh = color_thresh
        self.color_pad = color_pad
        self.dilate = dilate
        self.downsample = downsample

        # This is lower bound (in ms) of delay opencv waits for keyentry.
        self.key_delay = 1

        # Set up video writers
        self.writers = []

        # Register the variables we want to save to a json file later. This makes sure
        # we don't forget how we got a particular output.        
        self.register_metadata(
            "video_src",
            "min_area",
            "movement_thresh",
            "movement_pad",
            "color_thresh",
            "color_pad",
            "dilate",
            "downsample",
            "fourcc"
        )
        
    def reset_HSV(self, *args):
        self.low_H = 179
        self.low_S = 255
        self.low_V = 255
        self.high_H = 0
        self.high_S = 0
        self.high_V = 0

    def get_video_cap(self) -> ThreadedVideo:
        return cv2.VideoCapture(filename=self.video_src)
        
    def show_color_picker():
        """
        TODO: This should pop up a still from a video to let us pick the right
        pixels (and therefore HSV bounds) to select only flowers from a video.
        
        I think I want this such that if user manually passes in HSV values into
        __init__(), then we just use those, else we try to call this method (or
        just ignore it if we're not worried about color).

        And yes, color not colour just because opencv told me to.
        """
        pass

    @profile
    def display_video(self, show=True, write=False):    
        try:
            self.start_time = dt.now()

            # Get video capture
            video_cap = self.get_video_cap()
            self.video_cap = video_cap

            if write:
                # Make output sub-directory just for this run
                out_sub_dir = self.out_dir / str(self.start_time)
                Path.mkdir(out_sub_dir)

                # Get some metadata for videos
                fourcc = self.fourcc
                fps = self.fps
                width, height = self.shape

                # opencv only accepts string filepaths, not Path objects
                input_write_file = str(out_sub_dir / "input.avi") # for copy of input file, testing opencv encoding
                diff_write_file = str(out_sub_dir / "diff.avi") # for difference frames
                trail_write_file = str(out_sub_dir / "trail.avi") # for bee trail video
                
                # Create writers for whichever videos you want to output
                # input_video_writer = cv2.VideoWriter(filename=input_write_file, fourcc=self.fourcc, fps=fps, frameSize=(width, height))
                # diff_video_writer = cv2.VideoWriter(filename=diff_write_file, fourcc=self.fourcc, fps=fps, frameSize=(width, height))
                # trail_video_writer = cv2.VideoWriter(filename=trail_write_file, fourcc=self.fourcc, fps=fps, frameSize=(width, height))

                # input_video_writer = ThreadedVideoWriter(filename=input_write_file, fourcc=fourcc, fps=fps, frameSize=(width, height)).start()
                diff_video_writer = ThreadedVideoWriter(queue_size=256, flush_thresh=100, filename=diff_write_file, fourcc=self.fourcc, fps=fps, frameSize=(width, height))
                # trail_video_writer = ThreadedVideoWriter(filename=trail_write_file, fourcc=fourcc, fps=fps, frameSize=(width, height)).start()
            
            # Define windows here for clarity
            if show:
                cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
                cv2.namedWindow("Difference", cv2.WINDOW_NORMAL)
                cv2.namedWindow("Color", cv2.WINDOW_NORMAL)
                cv2.namedWindow("Trail", cv2.WINDOW_NORMAL)

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

                # Read next video frame, see if we've reached end
                frame_grabbed, orig_frame = video_cap.read()
                if not frame_grabbed:
                    break

                # Resize frame if required
                if self.downsample < 1:
                    frame = orig_frame.copy()
                    frame = cv2.resize(frame, dsize=None, fx=self.downsample, fy=self.downsample)
                else:
                    frame = orig_frame

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

                # Figure out upsample ratio
                if self.downsample < 1:
                    upsample_x = int(width / frame.shape[1])
                    upsample_y = int(height / frame.shape[0])
                    diff_mask = diff_mask.repeat(upsample_x, axis=1).repeat(upsample_y, axis=0)
                    color_mask = color_mask.repeat(upsample_x, axis=1).repeat(upsample_y, axis=0)

                # Convert masks to actual frames containing masked pixel data
                diff_frame = np.zeros(orig_frame.shape, np.uint8)
                diff_frame[diff_mask] = orig_frame[diff_mask]
                color_frame = np.zeros(orig_frame.shape, np.uint8)
                color_frame[color_mask] = orig_frame[color_mask]

                # Update trail frame too
                if trail_frame is None:
                    trail_frame = diff_frame.copy()
                    trail_updates = np.zeros(trail_frame.shape, dtype=bool)
                    trail_updates[diff_mask] = True
                else:
                    # Double-check that none of these pixels have been updated before
                    if not np.any(trail_updates, where=diff_mask):
                        trail_frame[diff_mask] = orig_frame[diff_mask]
                        trail_updates[diff_mask] = True
                
                # Copy normal frame so we can add frame number without messing up original frame
                if show:
                    i = self.current_frame_index
                    cv2.putText(frame_copy, f"Frame {i}/{total_frames}", (50, 100), self.FONT, 2, (0, 0, 255), 3)
                    frame_copy

                    # Show frames of interest
                    cv2.imshow("Input", frame_copy)
                    cv2.imshow("Difference", diff_frame)
                    cv2.imshow("Color", color_frame)
                    cv2.imshow("Trail", trail_frame)

                # Update previous frame now
                prev_frame = frame

                if write:
                    diff_video_writer.write(diff_frame)
                    # input_video_writer.write(frame)
                    # diff_video_writer.write(diff_frame)
                    # trail_video_writer.write(trail_frame)
                    # print(f"Queue size: {trail_video_writer.Q.qsize()}")utou

                # Respond to key press
                if show:
                    if key == ord("q"):
                        break
        except Exception as e:
            print(e)
        finally:
            self.end_time = dt.now()
            self.total_time = self.end_time - self.start_time
            print(f"Took {self.total_time.seconds} seconds")

            print("Releasing video capture")
            video_cap.release()
            print("Destroying all windows")
            cv2.destroyAllWindows()

            if write:
                print("Releasing video writers")
                # input_video_writer.release()
                diff_video_writer.release()
                # trail_video_writer.release()
                print("Saving metadata to file")
                # Meta file
                meta_fp = out_sub_dir / "meta.json"
                self.save_metadata(fp=meta_fp)

    def register_metadata(self, *keys):
        self._meta_keys = keys

    def save_metadata(self, fp):
        meta_d = {}
        # Use this loop to deal with possible missing values
        for key in self._meta_keys:
            try:
                val = self.__getattribute__(key)
            except AttributeError:
                val = "Not found"
            meta_d[key] = val

        # TODO: Think about if I really want to update these values here, or just
        # put them in __init__() with the rest (which is possible, but feels wrong).
        meta_d.update({
            "time_taken": str(self.total_time),
            "start_datetime": str(self.start_time),
        })

        # Sort alphabetically because it's easier to read
        meta_d = dict(sorted(meta_d.items()))

        with open(fp, "w") as f:
            json.dump(meta_d, fp=f)

    def get_video_prop(self, prop: int) -> int:
        """
        Helper method to get video capture properties as integers, defaulting
        to -1 if video capture doesn't exist.
        """
        if self.video_cap is not None:
            return int(self.video_cap.get(prop))
        return -1        

    @property
    def current_frame_index(self) -> int:
        return self.get_video_prop(cv2.CAP_PROP_POS_FRAMES)

    @property
    def total_frames(self) -> int:
        return self.get_video_prop(cv2.CAP_PROP_FRAME_COUNT)

    @property
    def fps(self) -> int:
        return self.get_video_prop(cv2.CAP_PROP_FPS)

    @property
    def shape(self) -> Tuple[int, int]:
        """Returns (width, height) of input video. Returns (-1, -1) if no video."""
        width = self.get_video_prop(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.get_video_prop(cv2.CAP_PROP_FRAME_HEIGHT)
        return width, height

    @profile
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
        if color is not None:
            gray = cv2.cvtColor(frame, color)
        else: # sometimes we use an already-gray input
            gray = frame.copy()
        _, threshed = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        
        # Only dilate if some kernel size was given
        if self.dilate is not None:
            dilate_kernel = np.ones((self.dilate, self.dilate))
            dilated = cv2.dilate(threshed, dilate_kernel)
        else:
            dilated = threshed

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


video_src = config("INPUT_FILEPATH")
out_dir = config("OUTPUT_DIRECTORY")
D = Display(video_src=video_src, out_dir=out_dir, movement_pad=5, movement_thresh=80, dilate=None, min_area=0, downsample=1)
D.display_video(write=True, show=False)