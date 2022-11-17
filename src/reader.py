from queue import Empty, Queue
from threading import Thread
from typing import Union
import time

import cv2


class ThreadedVideo:
    """
    Use this to perform read operations on a different thread to eliminate blocking
    times where possible.

    Stolen from https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/.
    """
    # This is the proportion of the max queue size that should be filled before triggering
    # the reading thread to sleep for a little bit in order to let the consumer
    # use up the current frames in the queue. Prevents overhead that results from
    # the queue filling up entirely!
    SLEEP_THRESH = 0.75

    def __init__(self, source: Union[str, int], queue_size:int=512):
        self.source = source
        self.queue_size = queue_size

        # Check if reading from live camera or video file
        if type(source) == str:
            self.stream = cv2.VideoCapture(filename=source)
        elif type(source) == int:
            self.stream = cv2.VideoCapture(index=source)
        else:
            raise ValueError(f"Expected `source` to be string or int, but got {type(source)}")
        
        self.stopped = False
        self.Q = Queue(maxsize=queue_size)
        self.queue_size = queue_size
        self.t = None  # To keep track of this Thread

    def start(self):
        if self.t is not None:
            raise RuntimeError("ThreadedVideo.start() was called more than once")

        # start a thread to read frames from the file video stream
        self.t = Thread(target=self.update)
        # self.t.daemon = True
        self.t.start()
        return self

    def get(self, key):
        try:
            return self.stream.get(key)
        except:
            return -1

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

            if self.Q.qsize() > self.SLEEP_THRESH * self.queue_size:
                print("Reader sleeping... (ignore if not using webcam input, otherwise this is very bad)")
                time.sleep(10)
            
            if not self.Q.full():
                grabbed, frame = self.stream.read()
                
                if not grabbed:
                    print("Reader is releasing itself!")
                    self.release()
                
                self.Q.put(frame)

    def read(self):
        try:
            return True, self.Q.get(timeout=10) # TODO: Think of more robust way of
        except Empty:
            return False, None
        

    def more(self):
        return self.Q.qsize() > 0

    def release(self):
        # Set this attribute and let the actual thread stop itself to make sure 
        # it exits cleanly
        self.stopped = True

    def __repr__(self) -> str:
        return f"<Reader: source=${self.source}>"
