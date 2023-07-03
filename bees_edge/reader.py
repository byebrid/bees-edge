from threading import Event
from queue import Queue
from typing import Union, Tuple
from logging import Logger
import time

import cv2

from bees_edge.logging_thread import LoggingThread


class Reader(LoggingThread):
    """A class to read video from either a file or camera, and push to a queue.
    
    Reads video from file or camera and pushes each frame onto a queue. Note that
    this queue *must* be emptied before the the program can be closed, meaning
    every frame in this queue needs to be handled by some other thread, either
    with some actual processing or just popping those frames and doing nothing 
    with them.

    This Reader includes "smart sleeping". This is intended for use with video 
    files, where there is no real upper bound on the rate at which we read frames
    (as opposed to a live camera feed, where you'll be bounded by its FPS). For 
    files, this Reader can read frames much more quickly than they can be 
    processed, meaning the reading queue can fill up and start blocking this 
    reading thread when it tries to push frames onto the full queue (and this 
    blocking still uses the CPU meaning it's a waste of compute time!). We 
    therefore let this thread sleep for a while once its queue fills above some
    threshold (e.g. ~90%).
    """
    def __init__(
        self,
        reading_queue: Queue,
        video_source: Union[str, int],
        stop_signal: Event,
        logger: Logger,
    ) -> None:
        """Initialise Reader with given queue and video source.

        Parameters
        ----------
        reading_queue : Queue
            The queue to push video frames onto.
        video_source : Union[str, int]
            A string representing a video file's filepath, or a non-negative 
            integer index for an attached camera device. 0 is usually your 
            laptop/rapsberry pi's in-built webcam, but this will depend on the 
            hardware you're using.
        stop_signal : Event
            A threading Event that this Reader queries to know when to stop. 
            This is used for graceful termination of the multithreaded program.
        logger : logging.Logger
            Logger to use for logging key info, warnings, etc.
        """
        super().__init__(name="ReaderThread", logger=logger)

        self.reading_queue = reading_queue
        self.video_source = video_source
        self.stop_signal = stop_signal

        # Make video capture now so we can dynamically retrieve its FPS and frame size
        try:
            self.vc = self.get_video_capture(source=self.video_source)
        except ValueError:
            self.stop_signal.set()
            self.reading_queue.put(None)
            self.error(f"Could not make VideoCapture from source '{video_source}'")

    def run(self) -> None:
        frame_count = 0

        while True:
            if self.stop_signal.is_set():
                self.info("Received stop signal")
                break

            grabbed, frame = self.vc.read()
            if not grabbed or frame is None:
                break

            self.reading_queue.put(frame)
            frame_count += 1
            if frame_count % 1000 == 0:
                self.info(f"Read {frame_count} frames so far")

        # Append None to indicate end of queue
        self.info("Adding None to end of reading queue")
        self.reading_queue.put(None)
        self.vc.release()
        self.stop_signal.set()

    def get_fps(self) -> int:
        return int(self.vc.get(cv2.CAP_PROP_FPS))

    def get_frame_size(self) -> Tuple[int]:
        width = int(self.vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    @staticmethod
    def get_video_capture(source: Union[str, int]) -> cv2.VideoCapture:
        """
        Get a VideoCapture object from either a given filepath or an interger
        representing the index of a webcam (e.g. source=0). Raises a ValueError if
        we could not create a VideoCapture from the given `source`.

        :param source: a string representing a filepath for a video, or an integer
            representing a webcam's index.
        :return: a VideoCapture object for the given `source`.
        """
        if type(source) is str:
            return cv2.VideoCapture(filename=source)
        elif type(source) is int:
            return cv2.VideoCapture(index=source)
        else:
            raise ValueError(
                "`source` must be a filepath to a video, or an integer index for the camera"
            )