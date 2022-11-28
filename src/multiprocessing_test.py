import time
from multiprocessing import Process, Queue

import cv2


def timeit(f):
    def wrapper():
        start = time.time()
        f()
        end = time.time()
        duration = end - start
        print(f"Finished in {duration:.2f} s")
    
    return wrapper


class SimpleReader(Process):
    def __init__(self, queue: Queue, filename: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = queue
        self.filename = filename

    def run(self):
        stream = cv2.VideoCapture(filename=self.filename)
        frame_count = 0

        try:
            while True:
                grabbed, frame = stream.read()
                if grabbed and frame is not None:
                    frame_count += 1
                    print(f"SimpleReader: Frame {frame_count}")
                    self.queue.put(frame)
                else:
                    break
        except KeyboardInterrupt:
            pass
        
        print("Adding DONE to end of queue")
        self.queue.put("DONE")
        print("SimpleReader: Releasing VideoCapture")
        stream.release()
        print("SimpleReader: Released VideoCapture")


class SimpleWriter(Process):
    def __init__(self, queue: Queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = queue

    def run(self):
        while True:
            frame = self.queue.get()
            print("SimpleWriter: Read frame")
            if frame == "DONE":
                print("SimpleWriter: Hit end of queue!")
                break


@timeit
def main():
    filename = "data/scaevola/scaevola.mp4"
    queue = Queue(maxsize=512)
    
    reader = SimpleReader(queue=queue, filename=filename)
    writer = SimpleWriter(queue=queue)

    print("Starting reader")
    reader.start()
    print("Starting writer")
    writer.start()
    print("Joining with reader")
    reader.join()
    print("Joining with writer")
    writer.join()

    # reader = ThreadedVideo(source="data/scaevola/scaevola.mp4")
    # print("Starting reader...")
    # reader.start()

    # i = 0
    # start = time.time()

    # try:
    #     while True:
    #         grabbed, frame = reader.read()
    #         if grabbed is False:
    #             break
    #         if frame is None:
    #             print("Frame is None!")

    #         i += 1
    #         if i % 1000 == 0:
    #             print(f"{i=}")
    # finally:
    #     print("Stopping reader...")
    #     reader.stop()
    
    # return i


if __name__ == "__main__":
    main()