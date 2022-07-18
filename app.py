import json
import string
from datetime import datetime as dt
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm
from windows.trail_window import TrailWindow
from windows.color_window import ColorWindow
from windows.window import Window
from windows.movement_window import MovementWindow
from windows.input_window import InputWindow

# Just to deal with @profile decorator from line_profiler. It's a weird system but whatever
if type(__builtins__) is not dict or "profile" not in __builtins__:
    profile = lambda f: f


def config(key, default=None):
    """Helper method to return value of `key` in `config.json`, else returns `default`"""
    with open("eg_config.json", "r") as f:
        d = json.load(f)
    return d.get(key, default)


KEYS = {
    "space": 32,
    "left": 91,  # Note this is actually "[", since left/right arrows are bugged in opencv with QT GUI (https://github.com/opencv/opencv/issues/20215)
    "right": 93,  # This is "]"
    "esc": 27,
    "none": 255,
}
# Add lowercase letters to dictionary as well so we better recognise user inputs
KEYS.update({l: ord(l) for l in string.ascii_lowercase})
# Keep reverse mapping to for when we know keycode, but want to know human-readable name
KEYCODE_TO_NAME = {v: k for k, v in KEYS.items()}


class App:
    def __init__(
        self,
        *,
        video_src: str,
        output_dir: str,
        output_ext: str,
        show: dict,
        write: dict,
        queue_size: int,
        flush_thresh: int,
        fourcc: int,
        frame_size: Tuple[int, int],
        movement_kwargs: dict,
        color_kwargs:dict
    ):
        self._video_src = video_src
        self._output_dir = output_dir
        self._output_ext = output_ext
        self._show = show
        self._write = write
        self._queue_size = queue_size
        self._flush_thresh = flush_thresh
        self._fourcc = fourcc
        self._frame_size = frame_size
        self._movement_kwargs = movement_kwargs
        self._color_kwargs = color_kwargs

        # Create video reader either for file or camera input
        if type(video_src) == str:
            self._reader = cv2.VideoCapture(filename=video_src)
        else: # assume int, cv2 will throw error for us anyway!
            self._reader = cv2.VideoCapture(video_src)
        # Get video parameters
        self._fps = fps = self.get_video_prop(cv2.CAP_PROP_FPS)
        self._width = self.get_video_prop(cv2.CAP_PROP_FRAME_WIDTH)
        self._height = self.get_video_prop(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_size = (self._width, self._height)

        # Set up Windows?
        self._input_win = InputWindow(
            show=show,
            write=write,
            output_ext=output_ext,
            queue_size=queue_size,
            flush_thresh=flush_thresh,
            fourcc=fourcc,
            fps=fps,
            frame_size=frame_size
        )
        self._movement_win = MovementWindow(
            show=show,
            write=write,
            output_ext=output_ext,
            queue_size=queue_size,
            flush_thresh=flush_thresh,
            fourcc=fourcc,
            fps=fps,
            frame_size=frame_size,
            thresh=movement_kwargs["THRESH"],
            dilate=movement_kwargs["DILATE"],
            pad=movement_kwargs["PAD"],
            min_area=movement_kwargs["MIN_AREA"]
        )
        self._color_win = ColorWindow(
            HSV_dict=color_kwargs["HSV_DICT"],
            show=show,
            write=write,
            output_ext=output_ext,
            output_dir=output_dir,
            queue_size=queue_size,
            flush_thresh=flush_thresh,
            fourcc=fourcc,
            fps=fps,
            frame_size=frame_size,
            thresh=color_kwargs["THRESH"],
            dilate=color_kwargs["DILATE"],
            pad=color_kwargs["PAD"],
            min_area=color_kwargs["MIN_AREA"]
        )
        self._trail_win = TrailWindow(
            show=show,
            write=write,
            output_ext=output_ext,
            output_dir=output_dir,
            queue_size=queue_size,
            flush_thresh=flush_thresh,
            fourcc=fourcc,
            fps=fps,
            frame_size=frame_size,
            in_window=self._movement_win
        )

        self._windows = [self._input_win, self._movement_win, self._color_win, self._trail_win] # type: List[Window]

        self.register_metadata(
            "video_src",
            "output_dir",
            "output_ext",
            "show",
            "write",
            "queue_size",
            "flush_thresh",
            "fourcc",
            "frame_size",
            "movement_kwargs",
            "color_kwargs"
        )

    @profile
    def start(self):
        try:
            self._start_time = dt.now()

            # We just care if we're showing/writing _any_ Windows
            write = any(self._write.values())
            show = any(self._show.values())

            if write:
                # Make output sub-directory just for this run
                out_sub_dir = Path(self._output_dir) / str(self._start_time)
                Path.mkdir(out_sub_dir, parents=True)

                # Make sure each window writes to this run's sub-directory
                for window in self._windows:
                    window.prepare_writer(output_dir=out_sub_dir)

            # # Trail frame is a cumulative frame of all movement detection frames.
            # trail_frame = None
            # # This is to help keep track of which pixels have already been set in trail frame. Doing this to try and get more bees
            # # rather than flowers in final image!
            # trail_updates = None

            # Play video, with all processing
            while True:
                if self.current_frame_index % 1000 == 0:
                    print(f"Frame {self.current_frame_index}/{self.total_frames}")

                if show:
                    # Check to see if user presses key
                    key = cv2.waitKey(1)
                    key &= 0xFF
                    if key == KEYS["q"]:
                        print("User pressed 'q', exiting...")
                        break
                    elif key != KEYS["none"]:
                        keyname = KEYCODE_TO_NAME.get(key, "unknown")
                        print(f"Keypress: {keyname}  (code={key})")

                # Read next video frame, see if we've reached end
                frame_grabbed, frame = self._reader.read()
                if not frame_grabbed:
                    break

                for window in self._windows:
                    window.update(frame=frame, index=self.current_frame_index)
                    # Window only shows/writes if it was told to originally, else does nothing!
                    # TODO: This might be confusing, maybe change it back to way it
                    # was before? Small issue is I want to say, if window.getShow(), then window.show(),
                    # but that's a little confusing, one show is attribute, other is method!
                    window.show()
                    window.write()

                # # Update trail frame too
                # if trail_frame is None:
                #     trail_frame = diff_frame.copy()
                #     trail_updates = np.zeros(trail_frame.shape, dtype=bool)
                #     trail_updates[diff_mask] = True
                # else:
                #     # Double-check that none of these pixels have been updated before
                #     if not np.any(trail_updates, where=diff_mask):
                #         trail_frame[diff_mask] = orig_frame[diff_mask]
                #         trail_updates[diff_mask] = True
        finally:
            self._end_time = dt.now()
            self._total_time = self._end_time - self._start_time
            print(f"Took {self._total_time.seconds} seconds")

            print("Releasing resources")
            self._reader.release()
            for window in self._windows:
                window.release()

            if write:
                # TODO: Maybe just copy config.json, but append extra data like time taken, etc.?
                meta_fp = out_sub_dir / "meta.json"
                # print(f"Copying config.json to {meta_fp}")
                self.save_metadata(fp=meta_fp)

    def register_metadata(self, *keys):
        self._meta_keys = keys

    def save_metadata(self, fp):
        meta_d = {}
        # Use this loop to deal with possible missing values
        for key in self._meta_keys:
            try:
                val = self.__getattribute__(f"_{key}") # I've been prepending underscores to attributes!
            except AttributeError:
                val = "Not found"
            meta_d[key] = val

        # TODO: Think about if I really want to update these values here, or just
        # put them in __init__() with the rest (which is possible, but feels wrong).
        meta_d.update(
            {
                "time_taken": str(self._total_time),
                "start_datetime": str(self._start_time),
            }
        )

        # Sort alphabetically because it's easier to read
        meta_d = dict(sorted(meta_d.items()))

        with open(fp, "w") as f:
            json.dump(meta_d, fp=f)

    def get_video_prop(self, prop: int) -> int:
        """
        Helper method to get video capture properties as integers, defaulting
        to -1 if video capture doesn't exist.
        """
        if self._reader is not None:
            return int(self._reader.get(prop))
        return -1

    @property
    def current_frame_index(self) -> int:
        return self.get_video_prop(cv2.CAP_PROP_POS_FRAMES)

    @property
    @lru_cache(maxsize=1)
    def total_frames(self) -> int:
        return self.get_video_prop(cv2.CAP_PROP_FRAME_COUNT)


# Reading from config
video_src = config("INPUT_FILEPATH")
show = config("SHOW", default=True)
write = config("WRITE", default=False)
output_dir = config("OUTPUT_DIR", default="out")
output_ext = config("OUTPUT_EXT", default=".avi")
queue_size = config("QUEUE_SIZE", default=100)
flush_thresh = config("FLUSH_THRESH", default=50)
fourcc = config("FOURCC", default="XVID")
frame_size = config("FRAME_SIZE", default=None)
movement_kwargs = config("MOVEMENT_KWARGS")
color_kwargs = config("COLOR_KWARGS")

app = App(
    video_src=video_src,
    show=show,
    write=write,
    output_dir=output_dir,
    output_ext=output_ext,
    queue_size=queue_size,
    flush_thresh=flush_thresh,
    fourcc=fourcc,
    frame_size=frame_size,
    movement_kwargs=movement_kwargs,
    color_kwargs=color_kwargs,
)
app.start()
