import json
import string
from datetime import datetime as dt
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple
import shutil

import cv2
from reader import ThreadedVideo
from windows.trail_window import TrailWindow
from windows.color_window import ColorWindow
from windows.window import Window
from windows.movement_window import MovementWindow
from windows.input_window import InputWindow
from fps import FPS

# Just to deal with @profile decorator from line_profiler. It's a weird system but whatever
if type(__builtins__) is not dict or "profile" not in __builtins__:
    profile = lambda f: f


def config(key, default=None):
    """Helper method to return value of `key` in `config.json`, else returns `default`"""
    # Get filepaths of true config, and example config files
    config_fp = Path("config.json")
    eg_config_fp = Path("eg_config.json")

    if not config_fp.exists():
        print(f"{config_fp} doesn't exist! Copying {eg_config_fp} to fix this. Edit {config_fp} in the future to adjust parameters")
        shutil.copy(eg_config_fp, config_fp)

    with open("config.json", "r") as f:
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
        input_width: int,
        input_height: int,
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
        self._input_width = input_width
        self._input_height = input_height
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
        self._reader = ThreadedVideo(source=video_src, queue_size=1024)

        # Override default resolution if specified
        if self._input_width is not None and self._input_height is not None:
            print(f"Manually setting input resolution to {self._input_width}x{self._input_height}")
            self._reader.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self._input_height)
            self._reader.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self._input_width)
        
        # Get video parameters
        self._fps = fps = self.get_video_prop(cv2.CAP_PROP_FPS)
        self._width = self.get_video_prop(cv2.CAP_PROP_FRAME_WIDTH)
        self._height = self.get_video_prop(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_size = (self._width, self._height)
        print(f"fps = {self._fps}; frame_size={frame_size}")

        # Set up Windows. When adding Window, make sure to also append it to
        # self._windows below!
        self._windows = []

        if self._show["Input"] or self._write["Input"]:
            self._windows.append(InputWindow(
                show=show,
                write=write,
                output_ext=output_ext,
                queue_size=queue_size,
                flush_thresh=flush_thresh,
                fourcc=fourcc,
                fps=fps,
                frame_size=frame_size
            ))
        if self._show["Movement"] or self._write["Movement"]:
            self._windows.append(MovementWindow(
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
            ))
        if self._show["Color"] or self._write["Color"]:
            self._windows.append(ColorWindow(
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
            ))
        if self._show["Trail"] or self._write["Trail"]:
            self._windows.append(TrailWindow(
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
            ))

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
        self._start_time = dt.now()
        
        # This lets us get average FPS for last 32 frames
        fps = FPS(size=32)

        try:
            # We just care if we're showing/writing _any_ Windows
            write = any(self._write.values())
            show = any(self._show.values())
            print(f"Writing to any files: {'Yes' if write else 'No'}")
            print(f"Showing any windows: {'Yes' if show else 'No'}")

            if write:
                # Make output sub-directory just for this run
                out_sub_dir = Path(self._output_dir) / str(self._start_time)
                Path.mkdir(out_sub_dir, parents=True)

                # Make sure each window writes to this run's sub-directory
                for window in self._windows:
                    window.prepare_writer(output_dir=out_sub_dir)

            # Start reading from camera/video file
            self._reader.start()

            # Play video, with all processing
            i = 1
            while True:
                # Show average FPS
                fps.tick()

                if i % 100 == 0:
                    print(f"Average FPS: {fps.get_average() if fps.get_average() is not None else -1:.2f}")
                    # Show queue size for debugging purposes
                    print(f"Frames currently in input/reading queue: {self._reader.Q.qsize()}")

                if i % 1000 == 0:
                    print(f"Frame {i}/{self.total_frames} ({i/self.total_frames * 100:.2f}%)")

                if show:
                    # Check to see if user presses key
                    key = cv2.waitKey(1)
                    key &= 0xFF
                    if key == KEYS["q"]:
                        print("User pressed 'q', exiting...")
                        break
                    elif key == KEYS["space"]:
                        input("Enter to continue")
                    elif key != KEYS["none"]:
                        keyname = KEYCODE_TO_NAME.get(key, "unknown")
                        print(f"Keypress: {keyname}  (code={key})")

                # Read next video frame, see if we've reached end
                frame_grabbed, frame = self._reader.read()
                if not frame_grabbed:
                    break

                for window in self._windows:
                    window.update(frame=frame, index=i)
                    # Window only shows/writes if it was told to originally, else does nothing!
                    # TODO: This might be confusing, maybe change it back to way it
                    # was before? Small issue is I want to say, if window.getShow(), then window.show(),
                    # but that's a little confusing, one show is attribute, other is method!
                    window.show()
                    window.write()

                i += 1
                
        finally:
            self._end_time = dt.now()
            self._total_time = self._end_time - self._start_time
            print(f"Took {self._total_time.seconds} seconds")

            print("Releasing camera/input file")
            self._reader.release()
            print("Releasing any output files, closing any windows")
            for window in self._windows:
                window.release()

            if write:
                print("Saving metadata to this run")
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


if __name__ == "__main__":
    # Reading from config
    video_src = config("INPUT_SOURCE")
    input_width = config("INPUT_WIDTH", default=None)
    input_height = config("INPUT_HEIGHT", default=None)
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
        input_width=input_width,
        input_height=input_height,
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
