from picamera2 import Picamera2
import cv2
import numpy as np

from fps import FPS

from threading import Thread, Condition, Event
from queue import Queue
from time import sleep
import datetime
from pprint import pprint


class PiReader(Thread):
    def __init__(self, hires_queue: Queue, diff_queue: Queue, threshold: int, min_moving_pixels:int=30, bitrate: int=10_000_000, hires_size: tuple[int, int]=(1280, 720), lores_size:tuple[int, int]=(640, 480), frame_guarantee_interval: int=30):
        super().__init__(name="PiReaderThread")

        self._hires_queue = hires_queue
        self._diff_queue = diff_queue

        self._threshold = threshold
        self._min_moving_pixels = min_moving_pixels
        self._bitrate = bitrate
        self._hires = {"size": hires_size, "format": "RGB888"}
        self._lores = {"size": lores_size, "format": "YUV420"}
        self._frame_guarantee_interval = frame_guarantee_interval
        
        self._camera = Picamera2()
        video_config = self._camera.create_video_configuration(main=self._hires, lores=self._lores)
        self._camera.configure(video_config)
        self._is_running = False
        self._use_lores = True

        # This is the upper index for getting the Y channel from a lores YUV
        # frame. Note that we just pick the first 'height' values to get
        # grayscale (https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf 6.1.1)
        self._lores_grayscale_index = self._lores["size"][1]

    def run(self):
        self._is_running = True
        self._camera.start()
        fps = FPS()

        self._prev_lores_frame = None
        self._frame_count = 0

        try:
            while self._is_running:
                fps.tick()
                print(f"{fps.get_average():.2f} FPS")
                self._process_frame()
                self._frame_count += 1
        finally:
            self._camera.stop()
            self._hires_queue.put(None)
            self._diff_queue.put(None)

    def _process_frame(self):
        # Ensure we occasionally provide full frame to output to keep track
        # of flowers, etc.
        frame_guarantee = self._frame_count % self._frame_guarantee_interval == 0

        lores_frame = self._get_frame(lores=True, gray=True)
        if self._prev_lores_frame is None:
            self._prev_lores_frame = lores_frame
            return

        abs_diff = cv2.absdiff(lores_frame, self._prev_lores_frame)
        movements = abs_diff > self._threshold
        if (movements.sum() >= self._min_moving_pixels) or frame_guarantee:
            abs_diff[movements] = 255
            abs_diff[~movements] = 0
            if not np.any(abs_diff) and not frame_guarantee:
                return

            # Read HIres frame into queue
            hires_frame = self._get_frame(lores=False)
            self._diff_queue.put(abs_diff)
            # Note that we add a boolean to indicate if hires frames needs to
            # written in full or not
            self._hires_queue.put((hires_frame, frame_guarantee))
        
        # Save current lores frame to compare with next frame
        self._prev_lores_frame = lores_frame

    def _get_frame(self, lores:bool=False, gray:bool=False):
        if lores:   
            stream_name = "lores"
        else:
            stream_name = "main"
        
        frame: np.ndarray = self._camera.capture_array(name=stream_name)
        # For now, we only allow grayscale for lores
        if gray and lores:
            frame = frame[:self._lores_grayscale_index]

        return frame

    def stop(self):
        self._is_running = False


class MotionDetector(Thread):
    QUEUE_TIMEOUT_SECONDS = 20

    def __init__(self, diff_queue: Queue, hires_queue: Queue, motion_detected_queue: list[Queue]):
        super().__init__(name="MotionDetectorThread")

        self._diff_queue = diff_queue
        self._hires_queue = hires_queue
        self._motion_detected_queue = motion_detected_queue

        self._dilation_kernel = np.ones((10, 10))
        self._is_running = False

    def run(self):
        self._is_running = True
        try:
            while self._is_running:
                self._process_frame()
        finally:
            self._motion_detected_queue.put(None)

    def _process_frame(self):
        # TODO: Fix queue timeout. This can timeout simply because there has
        # been no motion for a while
        diff_frame = self._diff_queue.get(timeout=self.QUEUE_TIMEOUT_SECONDS)
        hires_result = self._hires_queue.get(timeout=self.QUEUE_TIMEOUT_SECONDS)

        # Indicates the end of the queue. We should always have *both* of these being
        # None, not just one, since we always consume from both queues in the one
        # method call!
        # TODO: Consider adding check for only one being None since this is
        # unexpected behaviour
        if diff_frame is None or hires_result is None:
            self.stop()
            return

        # Note we expect tuple specifically for the hires queue
        hires_frame, frame_guarantee = hires_result
        
        # Guarantee that *full* frame is output every 'n' frames
        if frame_guarantee:
            self._motion_detected_queue.put(hires_frame)
            return

        mask = cv2.dilate(diff_frame, kernel=self._dilation_kernel)
        # Would like a cheaper way of upscaling this
        mask = cv2.resize(mask, dsize=(hires_frame.shape[1], hires_frame.shape[0]))
        mask = mask.astype(bool)

        # Return the final frame with only regions of "movement" included, everything
        # else blacked out
        motion_frame = np.zeros(shape=hires_frame.shape, dtype=np.uint8)
        motion_frame[mask] = hires_frame[mask]
        self._motion_detected_queue.put(motion_frame)

    def stop(self):
        self._is_running = False


class Writer(Thread):
    QUEUE_TIMEOUT_SECONDS = 10

    def __init__(self, file, size, fps: int, queue: Queue, fourcc: str="XVID"):
        super().__init__(name="WriterThread")

        self._file = file
        self._size = size
        self._fps = int(fps)
        self._queue = queue
        self._fourcc_str = fourcc  

        self._fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._is_running = False

    def run(self):
        self._is_running = True
        vw = cv2.VideoWriter(
            filename=str(self._file), # Ensure string instead of Path object
            fourcc=self._fourcc,
            fps=self._fps,
            frameSize=self._size,
        )

        try:
            while self._is_running:
                frame = self._queue.get(timeout=self.QUEUE_TIMEOUT_SECONDS)
                if frame is None:
                    break

                vw.write(frame)
        finally:
            vw.release()

    def stop(self):
        self._is_running = False


frame_guarantee_interval = 30
hires_size = (640, 480)
lores_size = (640, 480)
threshold = 50
min_moving_pixels = 30
queue_size = 128
fps = 30
timeout = datetime.timedelta(seconds=30)

output = f"out/{datetime.datetime.now().isoformat()}.avi"

hires_queue = Queue(maxsize=queue_size) 
diff_queue = Queue(maxsize=queue_size)
motion_detected_queue = Queue(maxsize=queue_size)

reader = PiReader(threshold=threshold, min_moving_pixels=min_moving_pixels, lores_size=lores_size, hires_size=hires_size, hires_queue=hires_queue, diff_queue=diff_queue, frame_guarantee_interval=frame_guarantee_interval)
motion_detector = MotionDetector(hires_queue=hires_queue, diff_queue=diff_queue, motion_detected_queue=motion_detected_queue)
writer = Writer(file=output, size=hires_size, fps=fps, queue=motion_detected_queue)

writer.start()
motion_detector.start()
reader.start()

try:
    print(f"Sleeping for {timeout.total_seconds()} seconds")
    sleep(timeout.total_seconds())
    print("Timed out")
finally:
    print("Releasing resources...")
    
    reader.stop()
    motion_detector.stop()
    writer.stop()

    reader.join()
    motion_detector.join()
    writer.join()
