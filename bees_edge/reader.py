from logging_thread import LoggingThread
from threading import Event
from queue import Queue
from typing import Union, Tuple
from logging import Logger
import time

import cv2


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
        sleep_seconds: int,
        flush_proportion: float,
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
        sleep_seconds : int
            *Initial* time to sleep. This sleep time will be dynamically updated
            according to smart sleep.
        flush_proportion : float
            When the queue fills above this proportion, trigger the sleeping to
            let other thread drain the queue. This prevents blocking when the
            queue becomes full. Recommended to be relatively high, e.g. 0.9.
        logger : logging.Logger
            Logger to use for logging key info, warnings, etc.
        """
        super().__init__(name="ReaderThread", logger=logger)

        self.reading_queue = reading_queue
        self.video_source = video_source
        self.stop_signal = stop_signal
        self.sleep_seconds = sleep_seconds

        self.flush_thresh = int(flush_proportion * reading_queue.maxsize)
        # Make video capture now so we can dynamically retrieve its FPS and frame size
        try:
            self.vc = self.get_video_capture(source=self.video_source)
        except ValueError:
            self.stop_signal.set()
            self.reading_queue.put(None)
            self.error(f"Could not make VideoCapture from source '{video_source}'")

        self.info(
            f"Will sleep {self.sleep_seconds} seconds if reading queue fills up with {self.flush_thresh} frames. This *should not happen* if you're using a live webcam, else the frames are being processed too slowly!"
        )

    def run(self) -> None:
        while True:
            if self.stop_signal.is_set():
                self.info("Received stop signal")
                break

            grabbed, frame = self.vc.read()
            if not grabbed or frame is None:
                break

            # Make sure queue has not filled up too much. This is really bad if
            # this happens for a *live* feed (i.e. webcam) since it means you
            # will lose the next `self.sleep_seconds` seconds of footage, but is
            # fine if working with video files as input
            if self.reading_queue.qsize() >= self.flush_thresh:
                self.debug(
                    f"Queue filled up to threshold. Sleeping {self.sleep_seconds} seconds to make sure queue can be drained..."
                )
                self.sleep_seconds = self.smart_sleep(
                    sleep_seconds=self.sleep_seconds, queue=self.reading_queue
                )
                self.debug(
                    f"Finished sleeping with {self.reading_queue.qsize()} frames still in buffer!"
                )

            self.reading_queue.put(frame)

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
        
    def smart_sleep(self, sleep_seconds: int, queue: Queue) -> int:
        """
        Sleeps for `sleep_seconds` seconds. Afterwards, if `queue` is completely
        empty, this implies that we slept for too long, in which case this will
        return a number slightly *smaller* than `sleep_seconds` which should be
        used for the next sleep. If `queue` is still too full, then this will
        return a *larger* number. If `queue` is in sweet spot, then just returns
        the given `sleep_seconds`.

        This will never return a negative number (so long as you don't provide it
        with a negative number).

        :param queue: queue whose throughput we want to optimise
        :return: adjusted sleep time
        """
        qsize_before = queue.qsize()
        time.sleep(sleep_seconds)
        self.debug("WOKE UP!")
        qsize_after = queue.qsize()

        lower_qsize = int(0.05 * queue.maxsize)
        upper_qsize = int(0.15 * queue.maxsize)

        # Best case, we don't need to adust sleep time
        if qsize_after >= lower_qsize and qsize_after <= upper_qsize:
            self.debug(
                f"Queue size: {qsize_before} -> {qsize_after}; within range of {lower_qsize}-{upper_qsize}; sleep of {sleep_seconds} sec was ideal!"
            )
            return sleep_seconds

        # If queue size is *really* small, then reduce sleep by large amount
        if qsize_after <= 2:
            new_sleep_seconds = sleep_seconds * 0.90
            self.debug(
                f"Queue size: {qsize_before} -> {qsize_after}; *below* range of {lower_qsize}-{upper_qsize}; adjusting sleep to {new_sleep_seconds:.2f} sec"
            )
            return new_sleep_seconds

        # Need to shorten sleep time if too many frames are drained
        if qsize_after < lower_qsize:
            new_sleep_seconds = sleep_seconds * 0.95
            self.debug(
                f"Queue size: {qsize_before} -> {qsize_after}; *below* range of {lower_qsize}-{upper_qsize}; adjusting sleep to {new_sleep_seconds:.2f} sec"
            )
            return new_sleep_seconds

        # Need to extend sleep time if too few frames are drained
        if qsize_after > upper_qsize:
            new_sleep_seconds = sleep_seconds * 1.05
            self.debug(
                f"Queue size: {qsize_before} -> {qsize_after}; *above* range of {lower_qsize}-{upper_qsize}; adjusting sleep to {new_sleep_seconds:.2f} sec"
            )
            return new_sleep_seconds