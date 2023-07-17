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
    def __init__(self, lores_queue, hires_queue, threshold: int, min_moving_pixels:int=30, bitrate: int=10_000_000, hires_size: tuple[int, int]=(1280, 720), lores_size:tuple[int, int]=(640, 480)):
        super().__init__(name="PiReaderThread")

        self._lores_queue = lores_queue
        self._hires_queue = hires_queue
        self._threshold = 40
        self._min_moving_pixels = min_moving_pixels
        self._bitrate = bitrate
        self._hires = {"size": hires_size, "format": "RGB888"}
        self._lores = {"size": lores_size, "format": "YUV420"}
        
        self._camera = Picamera2()
        video_config = self._camera.create_video_configuration(main=self._hires, lores=self._lores)
        self._camera.configure(video_config)
        self._is_running = False
        self._use_lores = True

    def run(self):
        self._is_running = True
        self._camera.start()
        fps = FPS()

        prev_lores_frame = None

        # lores is YUV colorspace, and first 'height' rows give the Y 
        # channel, the next height/4 rows contain the U channel and the 
        # final height/4 rows contain the V channel (https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf 6.1.1)
        lores_grayscale_index = self._lores["size"][1]

        try:
            # Ensure we get at least on hires frame so output video file is
            # non-empty
            self.disable_lores()
            hires_frame = self._get_frame()
            self._hires_queue.put(hires_frame)

            while self._is_running:
                fps.tick()
                print(f"{fps.get_average():.2f} FPS")
                
                # Read LOWres frame into queue
                self.enable_lores()
                lores_frame = self._get_frame()
                # Get only the Y values from this YUV array, i.e. make grayscale
                lores_frame = lores_frame[:lores_grayscale_index]

                if prev_lores_frame is not None:
                    abs_diff = cv2.absdiff(lores_frame, prev_lores_frame)
                    movements = abs_diff > self._threshold
                    if movements.sum() >= self._min_moving_pixels:
                        abs_diff[movements] = 255
                        abs_diff[~movements] = 0
                        if not np.any(abs_diff):
                            continue

                        # Read HIres frame into queue
                        self.disable_lores()
                        hires_frame = self._get_frame()
                        self._hires_queue.put(hires_frame)

                prev_lores_frame = lores_frame
        finally:
            self._camera.stop()

    def _get_frame(self):
        frame: np.ndarray = self._camera.capture_array(name=self._stream_name)
        return frame

    def stop(self):
        self._is_running = False

    def enable_lores(self):
        self._use_lores = True
        self._stream_name = "lores"
        self._width = self._lores["size"][0]
        self._height = self._lores["size"][1]

    def disable_lores(self):
        self._use_lores = False
        self._stream_name = "main"
        self._width = self._hires["size"][0]
        self._height = self._hires["size"][1]


# class MotionDetector(Thread):
#     def __init__(self, lores_queue: Queue, hires_queue: Queue):
#         super().__init__(name="MotionDetectorThread")

#         self._lores_queue = lores_queue
#         self._hires_queue = hires_queue

#         self._is_running = False

#     def run():
#         self._is_running = True
#         while self._is_running:
#             lores_frame = self._lores_queue.get()
#             # hires_frame = self._hires_queue.get()

#             if prev_lores_frame is None:
#                 # If processing very first frame, we have no prev to compare to,
#                 # so skip this iteration
#                 prev_lores_frame = lores_frame
#                 continue

#     def stop(self):
#         self._is_running = False


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

output = f"out/{datetime.datetime.now().isoformat()}.avi"

lores_queue = Queue(maxsize=512)
hires_queue = Queue(maxsize=512)
hires_size = (1280, 720)
lores_size = (640, 480)

reader = PiReader(threshold=200, hires_size=hires_size, hires_queue=hires_queue, lores_queue=lores_queue, lores_size=lores_size)
writer = Writer(file=output, size=hires_size, fps=30, queue=hires_queue)

writer.start()
reader.start()

timeout = datetime.timedelta(seconds=5)
try:
    sleep(timeout.seconds)
finally:
    print("Releasing resources...")
    
    reader.stop()
    writer.stop()

    reader.join()
    writer.join()
