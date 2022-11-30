from multiprocessing import Queue, Process, Event
from queue import Empty
from random import randint
import time
import signal
from typing import Union

import cv2


def producer(queue: Queue, source: Union[str, int], interrupt: Event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    if type(source) is str:
        vc = cv2.VideoCapture(filename=source)
    elif type(source) is int:
        vc = cv2.VideoCapture(index=source)
    else:
        raise ValueError("`source` must be a filepath to a video, or an integer index for the camera")

    print("Producer: STARTED!", flush=True)
    i = 0
    while True:
        if interrupt.is_set():
            print("Producer: Interrupted!", flush=True)
            break

        grabbed, frame = vc.read()
        if not grabbed or frame is None:
            break

        print(f"Producer: putting item#{i} onto queue", flush=True)
        queue.put(frame)
        i += 1
    
    print("Producer: Putting None on end of queue", flush=True)
    queue.put(None)
    print("Producer: FINISHED!", flush=True)
    vc.release()


def consumer(queue: Queue, filename: str, interrupt: Event, frameSize=(1920, 1080), fps: int = 60):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(filename=filename, fourcc=fourcc, fps=fps, frameSize=frameSize)

    print("Consumer: STARTED!", flush=True)
    i = 0
    while True:
        print(f"Consumer: Waiting for item...", flush=True)
        try:
            frame = queue.get(timeout=10)
        except Empty:
            print(f"Consumer: Waited too long for item! Exiting...", flush=True)
            break
        print(f"Consumer: Got item#{i}", flush=True)
        if frame is None:
            print("Consumer: Hit end of queue", flush=True)
            break
        
        vw.write(frame)
        i += 1
    
    print("Consumer: FINISHED!", flush=True)
    vw.release()


def ferry(producer_queue: Queue, consumer_queue: Queue, interrupt: Event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    i = 0
    while True:
        print(f"ferry: Getting frame {i} from producer...", flush=True)
        frame = producer_queue.get(timeout=5)

        print(f"ferry: Putting frame {i} into consumer...", flush=True)
        consumer_queue.put(frame)

        if frame is None:
            print(f"ferry: Found frame {i} was None!", flush=True)
            break

        cv2.imshow("Input", frame)
        cv2.waitKey(1)

        i += 1

if __name__ == "__main__":
    producer_queue = Queue(maxsize=512)
    consumer_queue = Queue(maxsize=512)

    interrupt_event = Event()
    consumer_done_event = Event()

    input_source = 0
    # input_filename = "data/scaevola/scaevola_long.mp4"
    output_filename = "test_out.avi"
    frame_size = (640, 480)
    fps = 30

    producer_proc = Process(target=producer, args=(producer_queue, input_source, interrupt_event), name="ProducerProc")
    ferry_proc = Process(target=ferry, args=(producer_queue, consumer_queue, interrupt_event))
    consumer_proc = Process(target=consumer, args=(consumer_queue, output_filename, interrupt_event, frame_size, fps), name="ConsumerProc")

    consumer_proc.start()
    producer_proc.start()
    ferry_proc.start()

    while True:
        try:
            time.sleep(0.5)
            if not consumer_proc.is_alive() and not producer_proc.is_alive() and not ferry_proc.is_alive():
                print("main: All child processes appear to have finished? Breaking out of infinite loop...")
                break
        except KeyboardInterrupt:
            print("main: Received KeyboardInterrupt. Setting interrupt event and breaking out of infinite loop...", flush=True)
            print("main: You may have to wait a minute for all child processes to gracefully exit!", flush=True)
            interrupt_event.set()
            break

    producer_proc.join()
    ferry_proc.join()
    consumer_proc.join()

    print(f"Producer queue: {producer_queue.qsize()} items", flush=True)
    print(f"Consumer queue: {consumer_queue.qsize()} items", flush=True)

