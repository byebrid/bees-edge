from __future__ import annotations
from threading import Thread, Event
import threading
from queue import Empty, Queue
import time
from typing import Union, Tuple
from datetime import datetime
from pathlib import Path
import json
import shutil

import cv2
import numpy as np

from fps import FPS


def config(key: str):
    with open("config.json", "r") as f:
        config_dict = json.load(f)
        return config_dict[key]


class PrintableThread(Thread):
    def __init__(self, name: str, verbose: bool = False) -> None:
        super().__init__(name=name)

        self.verbose = verbose

    def print(self, *args, important: bool = False, **kwargs):
        if important or self.verbose:
            print(f"{self.name:<16}: ", *args, **kwargs, flush=True)


class Reader(PrintableThread):
    def __init__(
        self,
        reading_queue: Queue,
        video_source: Union[str, int],
        stop_signal: Event,
        sleep_seconds: int,
        flush_proportion: float,
        verbose: bool = False,
    ) -> None:
        super().__init__(name="ReaderThread", verbose=verbose)

        self.reading_queue = reading_queue
        self.video_source = video_source
        self.stop_signal = stop_signal
        self.sleep_seconds = sleep_seconds

        self.flush_thresh = int(flush_proportion * reading_queue.maxsize)
        # Make video capture now so we can dynamically retrieve its FPS and frame size
        self.vc = self.get_video_capture(source=self.video_source)

        self.print(
            f"Will sleep {self.sleep_seconds} seconds if reading queue fills up with {self.flush_thresh} frames. This *should not happen* if you're using a live webcam, else the frames are being processed too slowly!",
            important=True,
        )

    def run(self) -> None:
        while True:
            if self.stop_signal.is_set():
                self.print("Interrupted!", important=True)
                break

            # self.print(f"Frames in READING queue = {self.reading_queue.qsize()}")
            grabbed, frame = self.vc.read()
            if not grabbed or frame is None:
                break

            # Make sure queue has not filled up too much. This is really bad if
            # this happens for a *live* feed (i.e. webcam) since it means you
            # will lose the next `self.sleep_seconds` seconds of footage, but is
            # fine if working with video files as input
            if self.reading_queue.qsize() >= self.flush_thresh:
                self.print(
                    f"Queue filled up to threshold. Sleeping {self.sleep_seconds} seconds to make sure queue can be drained..."
                )
                time.sleep(self.sleep_seconds)
                self.print(
                    f"Finished sleeping with {self.reading_queue.qsize()} frames still in buffer!"
                )

            self.reading_queue.put(frame)

        # Append None to indicate end of queue
        self.print("Adding None to end of reading queue", important=True)
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


class Writer(PrintableThread):
    def __init__(
        self,
        writing_queue: Queue,
        filepath: str,
        frame_size: Tuple[int, int],
        fps: int,
        stop_signal: Event,
        verbose: bool = False,
    ) -> None:
        super().__init__(name="WriterThread", verbose=verbose)

        self.writing_queue = writing_queue
        self.filepath = filepath
        self.frame_size = frame_size
        self.fps = fps
        self.stop_signal = stop_signal

        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.flush_thresh = int(0.75 * writing_queue.maxsize)
        self.print(
            f"Will flush buffer to output file every {self.flush_thresh} frames",
            important=True,
        )

    @classmethod
    def from_reader(
        cls,
        reader: Reader,
        writing_queue: Queue,
        filepath: str,
        stop_signal: Event,
        verbose: bool = False,
    ) -> Writer:
        fps = reader.get_fps()
        frame_size = reader.get_frame_size()
        writer = Writer(
            writing_queue=writing_queue,
            filepath=filepath,
            frame_size=frame_size,
            fps=fps,
            stop_signal=stop_signal,
            verbose=verbose,
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
            if (
                self.writing_queue.qsize() >= self.flush_thresh
                or self.stop_signal.is_set()
            ):
                self.print(
                    f"Queue size exceeded ({self.writing_queue.qsize() >= self.flush_thresh}) OR stop signal ({self.stop_signal.is_set()})"
                )

                # Only flush the threshold number of frames, OR remaining frames if there are only a few left
                frames_to_flush = min(self.writing_queue.qsize(), self.flush_thresh)
                self.print(f"Flushing {frames_to_flush} frames...")

                for i in range(frames_to_flush):
                    try:
                        frame = self.writing_queue.get(timeout=10)
                    except Empty:
                        self.print(
                            f"Waited too long for frame! Exiting...", important=True
                        )
                        loop_is_running = False
                        break
                    if frame is None:
                        loop_is_running = False
                        break

                    vw.write(frame)
                self.print(f"Flushed {frames_to_flush} frames!")

            time.sleep(2)

        vw.release()


class Ferry(PrintableThread):
    def __init__(
        self,
        reading_queue: Queue,
        motion_input_queue: Queue,
        stop_signal: Event,
        sleep_seconds: int,
        verbose: bool = False,
    ) -> None:
        super().__init__(name="FerryThread", verbose=verbose)

        self.reading_queue = reading_queue
        self.motion_input_queue = motion_input_queue
        self.stop_signal = stop_signal

        self.sleep_seconds = sleep_seconds

    def run(self) -> None:
        fps = FPS()
        fps.tick()
        i = 1

        flush_thresh = int(0.9 * self.motion_input_queue.maxsize)
        while True:
            try:
                frame = self.reading_queue.get(timeout=5)
                fps.tick()
            except Empty:
                self.print(
                    "Timed out waiting for frame from reading queue! Perhaps consider changing how long the Reader sleeps for?",
                    important=True,
                )
                continue

            if self.motion_input_queue.qsize() >= flush_thresh:
                self.print(
                    f"    WARNING - Sleeping for {self.sleep_seconds} seconds to let MotionDetector clear its queue...",
                    important=True,
                )
                time.sleep(self.sleep_seconds)
                self.print("Finished sleeping!")

            self.motion_input_queue.put(frame)

            if frame is None:
                return

            if i % 1000 == 0:
                self.print(f"handled frame#{i}")
                self.print(f"FPS = {fps.get_average()}", important=True)

            i += 1


class MotionDetector(PrintableThread):
    def __init__(
        self,
        input_queue: Queue,
        writing_queue: Queue,
        downscale_factor: int,
        dilate_kernel_size: int,
        movement_threshold: int,
        persist_factor: float,
        stop_signal: Event,
        verbose: bool = False,
    ) -> None:
        super().__init__(name="MotionThread", verbose=verbose)

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

        self.print(f"Dilation kernel is {self.dilation_kernel.shape}", important=True)

    def run(self) -> None:
        self.prev_frame = None

        while True:
            try:
                frame = self.input_queue.get(timeout=10)
            except Empty:
                if self.stop_signal.is_set():
                    self.print(
                        "Waited too long to get frame from input queue? This shouldn't happen!",
                        important=True,
                    )
                continue
            if frame is None:
                break

            motion_detected_frame = self.detect_motion(frame=frame)
            self.writing_queue.put(motion_detected_frame)

        # Make sure motion writer knows to stop
        self.writing_queue.put(None)

    def detect_motion(self, frame):
        # Downscale input frame
        orig_shape = frame.shape

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
        mask = cv2.resize(mask, dsize=(orig_shape[1], orig_shape[0]))

        # Convert to boolean so we can actually use it as a mask now
        mask = mask.astype(bool)

        # Save downscaled frame for use in next iteration
        self.prev_frame = downscaled_frame
        self.prev_diff = diff

        # Return the final frame with only regions of "movement" included, everything
        # else blacked out
        motion_frame = np.zeros(shape=frame.shape, dtype=np.uint8)
        motion_frame[mask] = frame[mask]

        return motion_frame


def main(
    video_source: str,
    reader_sleep_seconds: float,
    reader_flush_proportion: float,
    downscale_factor: int,
    dilate_kernel_size: int,
    movement_threshold: int,
    persist_factor: float,
    num_opencv_threads: int,
):
    start = time.time()

    cv2.setNumThreads(num_opencv_threads)

    reading_queue = Queue(maxsize=512)
    motion_input_queue = Queue(maxsize=512)
    writing_queue = Queue(maxsize=512)

    stop_signal = Event()
    pause_reading_signal = Event()

    output_directory = Path(f"out/{datetime.now()}")
    output_directory.mkdir()
    output_filepath = str(output_directory / "motion001.avi")
    print(f"Outputting to {output_filepath}")

    threads = (
        reader := Reader(
            reading_queue=reading_queue,
            video_source=video_source,
            stop_signal=stop_signal,
            sleep_seconds=reader_sleep_seconds,
            flush_proportion=reader_flush_proportion,
            verbose=True,
        ),
        motion_detector := MotionDetector(
            input_queue=reading_queue,
            writing_queue=writing_queue,
            downscale_factor=downscale_factor,
            dilate_kernel_size=dilate_kernel_size,
            movement_threshold=movement_threshold,
            persist_factor=persist_factor,
            stop_signal=stop_signal,
            verbose=True,
        ),
        writer := Writer.from_reader(
            reader=reader,
            writing_queue=writing_queue,
            filepath=output_filepath,
            stop_signal=stop_signal,
            verbose=True,
        ),
    )

    for thread in threads:
        print(f"main: Starting {thread.name}")
        thread.start()

    while True:
        try:
            time.sleep(5)
            if not any([thread.is_alive() for thread in threads]):
                print(
                    "main: All child processes appear to have finished! Exiting infinite loop..."
                )
                break

            print("\n", flush=True)
            for queue, queue_name in zip(
                [reading_queue, motion_input_queue, writing_queue],
                ["Reading", "Motion", "Writing"],
            ):
                print(f"{queue_name} queue size: {queue.qsize()}")
            print("\n", flush=True)
        except (KeyboardInterrupt, Exception) as e:
            print(
                "main: Received KeyboardInterrupt or some kind of Exception. Setting interrupt event and breaking out of infinite loop...",
                flush=True,
            )
            print(
                "main: You may have to wait a minute for all child processes to gracefully exit!",
                flush=True,
            )
            print(e)
            stop_signal.set()
            break

    for thread in threads:
        print(f"main: Joining {thread.name}")
        thread.join()

    # Copy config for record-keeping
    shutil.copy("config.json", output_directory / "config.json")

    # Add any extra stats/metadata to output too
    end = time.time()
    duration_seconds = end - start
    stats = {"duration_seconds": round(duration_seconds, 2)}
    with open(output_directory / "stats.json", "w") as f:
        json.dump(stats, f)

    print(f"Finished main script in {duration_seconds:.2f} seconds.")


if __name__ == "__main__":
    main(
        video_source=config("video_source"),
        reader_sleep_seconds=config("reader_sleep_seconds"),
        reader_flush_proportion=config("reader_flush_proportion"),
        downscale_factor=config("downscale_factor"),
        dilate_kernel_size=config("dilate_kernel_size"),
        movement_threshold=config("movement_threshold"),
        persist_factor=config("persist_factor"),
        num_opencv_threads=config("num_opencv_threads"),
    )
