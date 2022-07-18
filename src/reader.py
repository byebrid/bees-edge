from queue import Queue

import cv2


class ThreadedVideo:
    """
    Use this to perform read operations on a different thread to eliminate blocking
    times where possible.

    Stolen from https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/.
    """

    def __init__(self, path, queue_size=512):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False

        self.Q = Queue(maxsize=queue_size)
        self.t = None  # To keep track of this Thread

    def start(self):
        # start a thread to read frames from the file video stream
        self.t = Thread(target=self.update)
        # t.daemon = True
        self.t.start()
        return self

    def get(self, key):
        return self.stream.get(key)

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                self.stream.release()
                return
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                grabbed, frame = self.stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
                # add the frame to the queue
                self.Q.put(frame)

    def read(self):
        # return next frame in the queue
        return True, self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def release(self):
        # indicate that the thread should be stopped
        self.stopped = True
