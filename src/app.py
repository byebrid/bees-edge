from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Tuple, Union, Optional
from itertools import product

import cv2
import numpy as np

from fps import FPS

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


class Config:
    """Configuration class just to provide better parameter hints in code editor."""
    def __init__(
        self,
        video_source: str,
        reader_sleep_seconds: float,
        reader_flush_proportion: float,
        downscale_factor: int,
        dilate_kernel_size: int,
        movement_threshold: int,
        persist_factor: float,
        num_opencv_threads: int,
    ) -> None:
        self.video_source = video_source
        self.reader_sleep_seconds = reader_sleep_seconds
        self.reader_flush_proportion = reader_flush_proportion
        self.downscale_factor = downscale_factor
        self.dilate_kernel_size = dilate_kernel_size
        self.movement_threshold = movement_threshold
        self.persist_factor = persist_factor
        self.num_opencv_threads = num_opencv_threads


# Create Config object from JSON file
with open("config.json", "r") as f:
    __config_dict = json.load(f)
    CONFIG = Config(**__config_dict)


class LoggingThread(Thread):
    """A wrapper around `threading.Thread` with convenience methods for logging."""
    def __init__(self, name: str, logger: logging.Logger) -> None:
        super().__init__(name=name)

        self.logger = logger

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)


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
        logger: logging.Logger,
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
            LOGGER.debug(
                f"Queue size: {qsize_before} -> {qsize_after}; within range of {lower_qsize}-{upper_qsize}; sleep of {sleep_seconds} sec was ideal!"
            )
            return sleep_seconds

        # If queue size is *really* small, then reduce sleep by large amount
        if qsize_after <= 2:
            new_sleep_seconds = sleep_seconds * 0.90
            LOGGER.debug(
                f"Queue size: {qsize_before} -> {qsize_after}; *below* range of {lower_qsize}-{upper_qsize}; adjusting sleep to {new_sleep_seconds:.2f} sec"
            )
            return new_sleep_seconds

        # Need to shorten sleep time if too many frames are drained
        if qsize_after < lower_qsize:
            new_sleep_seconds = sleep_seconds * 0.95
            LOGGER.debug(
                f"Queue size: {qsize_before} -> {qsize_after}; *below* range of {lower_qsize}-{upper_qsize}; adjusting sleep to {new_sleep_seconds:.2f} sec"
            )
            return new_sleep_seconds

        # Need to extend sleep time if too few frames are drained
        if qsize_after > upper_qsize:
            new_sleep_seconds = sleep_seconds * 1.05
            LOGGER.debug(
                f"Queue size: {qsize_before} -> {qsize_after}; *above* range of {lower_qsize}-{upper_qsize}; adjusting sleep to {new_sleep_seconds:.2f} sec"
            )
            return new_sleep_seconds


class Writer(LoggingThread):
    def __init__(
        self,
        writing_queue: Queue,
        filepath: str,
        frame_size: Tuple[int, int],
        fps: int,
        stop_signal: Event,
        logger: logging.Logger,
    ) -> None:
        super().__init__(name="WriterThread", logger=logger)

        self.writing_queue = writing_queue
        self.filepath = filepath
        self.frame_size = frame_size
        self.fps = fps
        self.stop_signal = stop_signal

        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.flush_thresh = int(0.75 * writing_queue.maxsize)
        self.info(
            f"Will flush buffer to output file every {self.flush_thresh} frames"
        )

        self.frame_count = 0

    @classmethod
    def from_reader(
        cls,
        reader: Reader,
        writing_queue: Queue,
        filepath: str,
        stop_signal: Event,
        logger: logging.Logger,
    ) -> Writer:
        """Convenience method to generate a Writer from a Reader.

        This is useful because the Writer should share the FPS and resolution
        of the input video as determined by the Reader. This just saves you
        having to parse those attributes yourself.

        Parameters
        ----------
        reader : Reader
            Reader whose 
        writing_queue : Queue
            Queue to retrieve video frames from. Some other thread should be 
            putting these frames into this queue for this Writer to retrieve.
        filepath : str
            Filepath for output video file.
        stop_signal : Event
            A threading Event that this Reader queries to know when to stop. 
            This is used for graceful termination of the multithreaded program.
        logger : logging.Logger
            Logger to use for logging key info, warnings, etc.

        Returns
        -------
        Writer
            Writer with same FPS and frame size as given Reader.
        """
        fps = reader.get_fps()
        frame_size = reader.get_frame_size()
        writer = Writer(
            writing_queue=writing_queue,
            filepath=filepath,
            frame_size=frame_size,
            fps=fps,
            stop_signal=stop_signal,
            logger=logger,
        )
        return writer

    def run(self) -> None:
        vw = cv2.VideoWriter(
            filename=self.filepath,
            fourcc=self.fourcc,
            fps=self.fps,
            frameSize=self.frame_size,
        )

        loop_is_running = True
        while loop_is_running:
            time.sleep(2)

            if (
                self.writing_queue.qsize() < self.flush_thresh
                and not self.stop_signal.is_set()
            ):
                continue

            self.debug(
                f"Queue size exceeded ({self.writing_queue.qsize() >= self.flush_thresh}) OR stop signal ({self.stop_signal.is_set()})"
            )

            # Only flush the threshold number of frames, OR remaining frames if there are only a few left
            frames_to_flush = min(self.writing_queue.qsize(), self.flush_thresh)
            self.debug(f"Flushing {frames_to_flush} frames...")

            for i in range(frames_to_flush):
                try:
                    frame = self.writing_queue.get(timeout=10)
                except Empty:
                    self.warning(f"Waited too long for frame! Exiting...")
                    loop_is_running = False
                    break
                if frame is None:
                    loop_is_running = False
                    break

                vw.write(frame)
                self.frame_count += 1

                if self.frame_count % 1000 == 0:
                    self.info(f"Written {self.frame_count} frames so far")
            self.debug(f"Flushed {frames_to_flush} frames!")

        vw.release()


class MotionDetector(LoggingThread):
    def __init__(
        self,
        input_queue: Queue,
        writing_queue: Queue,
        downscale_factor: int,
        dilate_kernel_size: int,
        movement_threshold: int,
        persist_factor: float,
        stop_signal: Event,
        logger: logging.Logger,
    ) -> None:
        super().__init__(name="MotionThread", logger=logger)

        self.input_queue = input_queue
        self.writing_queue = writing_queue
        self.stop_signal = stop_signal

        self.prev_frame = None
        self.prev_diff = None

        # For motion detection
        self.downscale_factor = downscale_factor
        self.fx = self.fy = 1 / downscale_factor
        downscaled_kernel_size = int(dilate_kernel_size / self.downscale_factor)
        self.dilation_kernel = np.ones((downscaled_kernel_size, downscaled_kernel_size))
        self.movement_threshold = movement_threshold
        self.persist_factor = persist_factor

        if self.downscale_factor != 1:
            self.info(
                f"Dilation kernel downscaled by {downscale_factor}x from {dilate_kernel_size} to {self.dilation_kernel.shape[0]}"
            )

    def run(self) -> None:
        self.prev_frame = None

        while True:
            try:
                frame = self.input_queue.get(timeout=10)
            except Empty:
                if self.stop_signal.is_set():
                    self.error(
                        "Waited too long to get frame from input queue? This shouldn't happen!"
                    )
                continue
            if frame is None:
                break

            motion_detected_frame = self.detect_motion(frame=frame)

            # Ignore frames that have *zero* movement in them
            if np.any(motion_detected_frame):
                self.writing_queue.put(motion_detected_frame)

        # Make sure motion writer knows to stop
        self.writing_queue.put(None)

    def detect_motion(self, frame):
        # Downscale input frame
        orig_shape = frame.shape

        if self.downscale_factor == 1:
            # Downscale factor of 1 really just means no downscaling at all
            downscaled_frame = frame
        else:
            downscaled_frame = cv2.resize(frame, dsize=None, fx=self.fx, fy=self.fy)

        if self.prev_frame is None:
            self.prev_frame = downscaled_frame
        if self.prev_diff is None:
            self.prev_diff = np.zeros(self.prev_frame.shape, dtype=np.uint8)

        # Compute pixel difference between consecutive frames (note this still has 3 channels)
        # Note that previous frame was already downscaled!
        diff = cv2.absdiff(downscaled_frame, self.prev_frame)
        # Add decayed version of previous diff to help temporarily stationary bees "persist"
        diff += (self.prev_diff * self.persist_factor).astype(np.uint8)

        # Convert to grayscale
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # Cut off pixels that did not have "enough" movement. This is now a 2D array
        # of just 1s and 0s
        _, threshed_diff = cv2.threshold(
            src=gray, thresh=self.movement_threshold, maxval=255, type=cv2.THRESH_BINARY
        )
        mask = cv2.dilate(threshed_diff, kernel=self.dilation_kernel)

        # Up-res the final mask (note opencv expects opposite order of dimensions because of course it does)
        if self.downscale_factor != 1:
            mask = cv2.resize(mask, dsize=(orig_shape[1], orig_shape[0]))

        # Convert to boolean so we can actually use it as a mask now
        mask = mask.astype(bool)

        # Save downscaled frame for use in next iteration
        self.prev_frame = downscaled_frame
        # Note that we don't save the thresholded diff here. Otherwise, movement
        # could only persist across single frames at most, which is no good!
        self.prev_diff = diff

        # Return the final frame with only regions of "movement" included, everything
        # else blacked out
        motion_frame = np.zeros(shape=frame.shape, dtype=np.uint8)
        motion_frame[mask] = frame[mask]

        return motion_frame


def main(config: Config):
    start = time.time()

    # Make sure opencv doesn't use too many threads and hog CPUs
    cv2.setNumThreads(config.num_opencv_threads)

    # Create queues for transferring data between threads (or processes)
    reading_queue = Queue(maxsize=512)
    motion_input_queue = Queue(maxsize=512)
    writing_queue = Queue(maxsize=512)

    stop_signal = Event()

    # Figure out output filepath for this particular run
    output_directory = Path(f"out/{datetime.now()}")
    output_directory.mkdir()
    output_filepath = str(output_directory / "motion001.avi")

    # Copy config for record-keeping
    with open(output_directory / "config.json", "w") as f:
        print(config.__dict__)
        json.dump(config.__dict__, f)

    # Create some handlers for logging output to both console and file
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(threadName)-14s] %(msg)s"))
    file_handler = logging.FileHandler(filename=output_directory / "output.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)-8s] [%(threadName)-14s] %(msg)s")
    )
    # Make sure any prior handlers are removed
    LOGGER.handlers.clear()
    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)

    LOGGER.info("Running main() with Config: ", config.__dict__)
    LOGGER.info(f"Outputting to {output_filepath}")

    # Create all of our threads
    threads = (
        reader := Reader(
            reading_queue=reading_queue,
            video_source=config.video_source,
            stop_signal=stop_signal,
            sleep_seconds=config.reader_sleep_seconds,
            flush_proportion=config.reader_flush_proportion,
            logger=LOGGER,
        ),
        motion_detector := MotionDetector(
            input_queue=reading_queue,
            writing_queue=writing_queue,
            downscale_factor=config.downscale_factor,
            dilate_kernel_size=config.dilate_kernel_size,
            movement_threshold=config.movement_threshold,
            persist_factor=config.persist_factor,
            stop_signal=stop_signal,
            logger=LOGGER,
        ),
        writer := Writer.from_reader(
            reader=reader,
            writing_queue=writing_queue,
            filepath=output_filepath,
            stop_signal=stop_signal,
            logger=LOGGER,
        ),
    )

    for thread in threads:
        LOGGER.info(f"Starting {thread.name}")
        thread.start()

    # Regularly poll to check if all threads have finished. If they haven't finished,
    # just sleep a little and check later
    while True:
        try:
            time.sleep(5)
            if not any([thread.is_alive() for thread in threads]):
                LOGGER.info(
                    "All child processes appear to have finished! Exiting infinite loop..."
                )
                break

            for queue, queue_name in zip(
                [reading_queue, motion_input_queue, writing_queue],
                ["Reading", "Motion", "Writing"],
            ):
                LOGGER.debug(f"{queue_name} queue size: {queue.qsize()}")
        except (KeyboardInterrupt, Exception) as e:
            LOGGER.exception(
                "Received KeyboardInterrupt or some kind of Exception. Setting interrupt event and breaking out of infinite loop...",
            )
            LOGGER.warning(
                "You may have to wait a minute for all child processes to gracefully exit!",
            )
            stop_signal.set()
            break

    for thread in threads:
        LOGGER.info(f"Joining {thread.name}")
        thread.join()

    # Add any extra stats/metadata to output too
    end = time.time()
    duration_seconds = end - start
    stats = {"duration_seconds": round(duration_seconds, 2)}
    with open(output_directory / "stats.json", "w") as f:
        json.dump(stats, f)

    LOGGER.info(f"Finished main() in {duration_seconds:.2f} seconds.")


if __name__ == "__main__":
    downscale_factor = CONFIG.downscale_factor
    dilate_kernel_size = CONFIG.dilate_kernel_size
    movement_threshold = CONFIG.movement_threshold
    persist_factor = CONFIG.persist_factor

    # Figure out if video is webcam index, single video file, or directory of
    # video files
    video_source = CONFIG.video_source
    if type(video_source) != int:
        video_source = Path(video_source)
        if video_source.is_dir():
            video_source = [str(v) for v in video_source.iterdir()]
        elif type(video_source) is not list:
            video_source = [str(video_source)]
    else:
        # Just to make it iterable
        video_source = [video_source]

    if type(downscale_factor) is not list:
        downscale_factor = [downscale_factor]
    if type(dilate_kernel_size) is not list:
        dilate_kernel_size = [dilate_kernel_size]
    if type(movement_threshold) is not list:
        movement_threshold = [movement_threshold]
    if type(persist_factor) is not list:
        persist_factor = [persist_factor]

    parameter_combos = product(
        video_source,
        downscale_factor,
        dilate_kernel_size,
        movement_threshold,
        persist_factor,
    )
    parameter_keys = [
        "video_source",
        "downscale_factor",
        "dilate_kernel_size",
        "movement_threshold",
        "persist_factor",
    ]
    for combo in parameter_combos:
        this_config_dict = dict(zip(parameter_keys, combo))
        this_config_dict.update(
            {
                "reader_sleep_seconds": CONFIG.reader_sleep_seconds,
                "reader_flush_proportion": CONFIG.reader_flush_proportion,
                "num_opencv_threads": CONFIG.num_opencv_threads,
            }
        )
        this_config = Config(**this_config_dict)
        main(this_config)
