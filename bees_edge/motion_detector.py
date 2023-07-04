from logging import Logger
from queue import Empty, Queue
from threading import Event

import cv2
import numpy as np

from bees_edge.logging_thread import LoggingThread


class MotionDetector(LoggingThread):
    """Class that applies motion detection to frames in a reading queue.
    
    This applies the simplistic motion detection algorithm to frames in the input
    queue, and pushes the modified frames to the writing queue.
    """
    def __init__(
        self,
        input_queue: Queue,
        writing_queue: Queue,
        downscale_factor: int,
        dilate_kernel_size: int,
        movement_threshold: int,
        persist_factor: float,
        stop_signal: Event,
        logger: Logger,
    ) -> None:
        super().__init__(name="MotionThread", logger=logger)

        self.input_queue = input_queue
        self.writing_queue = writing_queue
        self.stop_signal = stop_signal

        self.prev_gray = None
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
        self.prev_gray = None

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

        # Convert to grayscale
        gray = cv2.cvtColor(downscaled_frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray


        # Compute pixel difference between consecutive frames (note this still has 3 channels)
        diff = cv2.absdiff(gray, self.prev_gray)
        if self.prev_diff is None:
            self.prev_diff = np.zeros(diff.shape, dtype=np.uint8)
        # Add decayed version of previous diff to help temporarily stationary bees "persist"
        # Note the astype() is necessary, otherwise just get noise all over the frame
        diff += (self.prev_diff * self.persist_factor).astype(np.uint8)
        
        # Cut off pixels that did not have "enough" movement. This is now a 2D array
        # of just 1s and 0s
        _, threshed_diff = cv2.threshold(
            src=diff, thresh=self.movement_threshold, maxval=255, type=cv2.THRESH_BINARY
        )
        mask = cv2.dilate(threshed_diff, kernel=self.dilation_kernel)

        # Up-res the final mask (note opencv expects opposite order of dimensions because of course it does)
        if self.downscale_factor != 1:
            mask = cv2.resize(mask, dsize=(orig_shape[1], orig_shape[0]))

        # Convert to boolean so we can actually use it as a mask now
        mask = mask.astype(bool)

        # Save downscaled frame for use in next iteration
        self.prev_gray = gray
        # Note that we don't save the thresholded diff here. Otherwise, movement
        # could only persist across single frames at most, which is no good!
        self.prev_diff = diff

        # Return the final frame with only regions of "movement" included, everything
        # else blacked out
        motion_frame = np.zeros(shape=frame.shape, dtype=np.uint8)
        motion_frame[mask] = frame[mask]

        return motion_frame