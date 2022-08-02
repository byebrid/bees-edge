from multiprocessing.sharedctypes import Value
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import cv2


class PiVideoCapture:
    """
    """
    def __init__(self, resolution = (320, 240), framerate = 30):
        self._cv2_props = {
            cv2.CAP_PROP_FRAME_WIDTH: resolution[0],
            cv2.CAP_PROP_FRAME_HEIGHT: resolution[1],
            cv2.CAP_PROP_FPS: framerate
        }

        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
            format="bgr", use_video_port=True)

    def read(self):
        try:
            f = next(self.stream).array # Just ask camera's video stream for current frame
            self.rawCapture.truncate(0)
            return True, f
        except:
            return False, None

    def release(self):
        # TODO: Figure out way to stop picamera capture_continuous()?
        pass
        
    def get(self, key: int):
        try:
            return self._cv2_props[key]
        except KeyError:
            return -1