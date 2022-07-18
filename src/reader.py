from multiprocessing.sharedctypes import Value
from queue import Queue
from threading import Thread
from typing import Union

import cv2


class ThreadedVideo:
    """
    Use this to perform read operations on a different thread to eliminate blocking
    times where possible.

    Stolen from https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/.
    """

    def __init__(self, source: Union[str, int], queue_size:int=512):
        # Check if reading from live camera or video file
        if type(source) == str:
            self.stream = cv2.VideoCapture(filename=source)
        elif type(source) == int:
            self.stream = cv2.VideoCapture(index=source)
        else:
            raise ValueError(f"Expected `source` to be string or int, but got {type(source)}")
        
        self.stopped = False

        self.Q = Queue(maxsize=queue_size)
        self.t = None  # To keep track of this Thread

    def start(self):
        # start a thread to read frames from the file video stream
        self.t = Thread(target=self.update)
        self.t.daemon = True
        self.t.start()
        return self

    def get(self, key):
        return self.stream.get(key)

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            
            if not self.Q.full():
                grabbed, frame = self.stream.read()
                
                if not grabbed:
                    self.stop()
                    return
                
                self.Q.put(frame)

    def read(self):
        return True, self.Q.get()

    def more(self):
        return self.Q.qsize() > 0

    def release(self):
        # Set this attribute and let the actual thread stop itself to make sure 
        # it exits cleanly
        self.stopped = True
        # Make sure we let that thread finish before we terminate everything!
        self.t.join()
