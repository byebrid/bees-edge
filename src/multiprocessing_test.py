import time
from multiprocessing import Process, Queue, set_start_method, Manager, JoinableQueue
from queue import Empty

import cv2


def timeit(f):
    def wrapper():
        start = time.time()
        f_output = f()
        end = time.time()
        duration = end - start
        print(f"Finished in {duration:.2f} s")

        return duration, f_output
    
    return wrapper


class EndOfQueue:
    """Sentinel value to indicate that you have reached the end of a Queue."""
    pass


class SimpleReader(Process):
    def __init__(self, queue: Queue, filename: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = queue
        self.filename = filename
        self.frame_count: int = 0

    def run(self):
        print(f"SimpleReader: PID = {self.pid}")

        self.stream = cv2.VideoCapture(filename=self.filename)

        try:
            while True:
                grabbed, frame = self.stream.read()
                if grabbed and frame is not None:
                    self.frame_count += 1
                    print(f"SimpleReader: Read frame {self.frame_count}")
                    self.queue.put(frame)
                else:
                    break
        except KeyboardInterrupt:
            print("SimpleReader: RECEIVED KEYBOARDINTERRUPT!")
        finally:
            print("SimpleReader: Adding EndOfQueue to end of queue")
            self.queue.put(EndOfQueue())
            self.queue.task_done()

            print("SimpleReader: Releasing VideoCapture")
            self.stream.release()
            print("SimpleReader: Released VideoCapture")


class SimpleWriter(Process):
    def __init__(self, queue: Queue, filename: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = queue
        self.filename = filename
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.fps = 60
        self.frameSize = (1920, 1080)

        self.frame_count = 0

    def run(self):
        print(f"SimpleWriter: PID = {self.pid}")

        self.video = cv2.VideoWriter(filename=self.filename, fourcc=self.fourcc, fps=self.fps, frameSize=self.frameSize)
        
        try:
            while True:
                frame = self.queue.get()
                self.frame_count += 1
                print(f"SimpleWriter: Got frame {self.frame_count}")
                if type(frame) is EndOfQueue:
                    print("SimpleWriter: Hit end of queue!")
                    break

                self.video.write(frame)
                print(f"SimpleWriter: Wrote frame {self.frame_count}")
        except KeyboardInterrupt:
            print("SimpleWriter: RECEIVED KEYBOARDINTERRUPT!")
        finally:
            print("SimpleWriter: Releasing VideoWriter")
            self.video.release()
            print("SimpleWriter: Returning")
            return


def terminate_queue(queue: Queue):
    try:
        while True:
            item = queue.get(block=False)
    except Empty:
        pass
    
    print("QUEUE: Empty! Closing...")
    # queue.close()
    print("QUEUE: Closed! Joining...")
    # queue.join_thread()
    # print("QUEUE: Joined!")



@timeit
def main():
    manager = Manager()
    reading_queue = manager.Queue(maxsize=512)
    # writing_queue = manager.Queue(maxsize=512)
    writing_queue = JoinableQueue(maxsize=512)

    filename = "data/scaevola/scaevola_long.mp4"
    
    reader = SimpleReader(queue=reading_queue, filename=filename)
    # writer = SimpleWriter(queue=writing_queue, filename="test_out.avi")

    print("Starting reader")
    reader.start()
    # print("Starting writer")
    # writer.start()

    try:
        frame_count = 0
        while True:
            frame = reading_queue.get()
            frame_count += 1
            
            print(f"Processing frame {frame_count}")
            # print(f"Putting frame {frame_count} onto writing queue")
            writing_queue.put(frame)
            writing_queue.task_done()

            if type(frame) is EndOfQueue:
                break
    except KeyboardInterrupt:
        print("main(): RECEIVED KEYBOARDINTERRUPT!")
        # print("Cleaning up reading queue")
        # terminate_queue(reading_queue)
        # print("Cleaning up writing queue")
        # terminate_queue(writing_queue)
    finally:
        writing_queue.join()
        while True:
            try:
                writing_queue.get(block=True, timeout=0.1)
            except Empty:
                break

        print("Joining with reader")
        reader.join()
        # print("Joining with writer")
        # writer.join()
        print("Finished joining")
        return frame_count - 1 # -1 because the EndOfQueue is counted above


if __name__ == "__main__":
    # set_start_method("spawn")
    (duration, frame_count) = main()
    print(f"Processed {frame_count} frames in {duration:.2f} seconds @ {frame_count / duration: .2f} FPS")