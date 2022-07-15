from queue import Queue
from threading import Thread

import cv2


class ThreadedVideoWriter:
    def __init__(self, filename: str, fourcc: int, fps: int, frameSize: int, queue_size: int, flush_thresh: int, **kwargs):
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
            raise ValueError(
                f"flush_thresh ({flush_thresh}) should be considerable less than queue_size {queue_size}!"
            )

        self.Q = Queue(maxsize=queue_size)
        self.flush_thresh = flush_thresh
        self.t = None  # to keep track of current writing Thread
        self.video = cv2.VideoWriter(filename=filename, fourcc=fourcc, fps=fps, frameSize=frameSize)
        self.dumping = False  # to keep track of whether we are currently writing to file or not

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
            if (
                wait
            ):  # let old thread block main thread so it can finish before starting our next one.
                self.t.join()
            else:
                raise ValueError("Tried to dump but previous thread was still running!")
        self.t = Thread(target=self.dump)
        self.t.start()

        if block:
            self.t.join()
