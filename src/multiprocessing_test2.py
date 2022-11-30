from multiprocessing import Queue, Process, Event
from queue import Empty
from random import randint
import time
import signal
from typing import Union

import cv2
import numpy as np

from utils import detect_motion


def get_video_capture(source: Union[str, int]) -> cv2.VideoCapture:
    """
    Get a VideoCapture object from either a given filepath or an interger 
    representing the index of a webcam (e.g. source=0). Raises a ValueError if
    we could not create a VideoCapture from the given `source`.

    :param source: a string representing a filepath for a video, or an integer
        representing a webcam's index.
    :return: a VideoCapture object for the given `source`.
    """
    if type(source) is str:
        return cv2.VideoCapture(filename=source)
    elif type(source) is int:
        return cv2.VideoCapture(index=source)
    else:
        raise ValueError("`source` must be a filepath to a video, or an integer index for the camera")



def reader(queue: Queue, source: Union[str, int], interrupt: Event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    vc = get_video_capture(source=source)

    print("Reader: STARTED!", flush=True)
    i = 0
    while True:
        if interrupt.is_set():
            print("Reader: Interrupted!", flush=True)
            break

        grabbed, frame = vc.read()
        if not grabbed or frame is None:
            break

        print(f"Reader: putting frame#{i} onto queue", flush=True)
        queue.put(frame)
        i += 1
    
    print("Reader: Putting None on end of queue", flush=True)
    queue.put(None)
    print("Reader: FINISHED!", flush=True)
    vc.release()


def writer(queue: Queue, filename: str, interrupt: Event, frameSize=(1920, 1080), fps: int = 60):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(filename=filename, fourcc=fourcc, fps=fps, frameSize=frameSize)

    print("Writer: STARTED!", flush=True)
    i = 0
    while True:
        print(f"Writer: Waiting for frame...", flush=True)
        try:
            frame = queue.get(timeout=10)
        except Empty:
            print(f"Writer: Waited too long for frame! Exiting...", flush=True)
            break
        print(f"Writer: Got frame#{i}", flush=True)
        if frame is None:
            print("Writer: Hit end of queue", flush=True)
            break
        
        vw.write(frame)
        i += 1
    
    print("Writer: FINISHED!", flush=True)
    vw.release()


def ferry(reader_queue: Queue, input_write_queue: Queue, motion_queue: Queue, interrupt: Event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    i = 0
    while True:
        print(f"ferry: Getting frame {i} from reader queue...", flush=True)
        frame = reader_queue.get(timeout=5)

        print(f"ferry: Putting frame {i} into input write queue...", flush=True)
        input_write_queue.put(frame)
        print(f"ferry: Putting frame {i} into motion queue...", flush=True)
        motion_queue.put(frame)

        if frame is None:
            print(f"ferry: Found frame {i} was None!", flush=True)
            break

        i += 1


def motion(motion_queue: Queue, writing_queue: Queue, interrupt: Event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    i = 0
    prev_frame = None
    while True:
        print(f"Motion: Getting frame {i} from motion queue...", flush=True)
        frame = motion_queue.get(timeout=5)
        # Initialise previous frame
        if prev_frame is None:
            prev_frame = frame

        if frame is None:
            print(f"Motion: Found frame {i} was None!", flush=True)
            # Make sure motion writer knows to stop
            writing_queue.put(None)
            break

        motion_mask = detect_motion(frame=frame, prev_frame=prev_frame)
        diff_frame = np.zeros(frame.shape, dtype=np.uint8)
        diff_frame[motion_mask] = frame[motion_mask]

        print(f"Motion: Putting frame {i} into motion write queue...", flush=True)
        writing_queue.put(diff_frame)

        prev_frame = frame
        i += 1

if __name__ == "__main__":
    reader_queue = Queue(maxsize=512)
    write_input_queue = Queue(maxsize=512)
    motion_queue = Queue(maxsize=512)
    write_motion_queue = Queue(maxsize=512)

    interrupt_event = Event()

    input_source = 0
    # input_source = "data/scaevola/scaevola_long.mp4"
    output_filename = "test_out.avi"
    frame_size = (640, 480)
    fps = 30

    reader_proc = Process(target=reader, args=(reader_queue, input_source, interrupt_event), name="ReaderProc")
    ferry_proc = Process(target=ferry, args=(reader_queue, write_input_queue, motion_queue, interrupt_event), name="FerryProc")
    write_input_proc = Process(target=writer, args=(write_input_queue, "input.avi", interrupt_event, frame_size, fps), name="WriteInputProc")
    motion_proc = Process(target=motion, args=(motion_queue, write_motion_queue, interrupt_event), name="MotionProc")
    write_motion_proc = Process(target=writer, args=(write_motion_queue, "motion.avi", interrupt_event, frame_size, fps), name="WriteMotionProc")

    procs = (
        reader_proc,
        ferry_proc,
        write_input_proc,
        motion_proc,
        write_motion_proc
    )

    for proc in procs:
        print(f"main: Starting {proc.name}")
        proc.start()

    while True:
        try:
            time.sleep(0.5)
            if not any([proc.is_alive() for proc in procs]):
                print("main: All child processes appear to have finished? Breaking out of infinite loop...")
                break
        except KeyboardInterrupt:
            print("main: Received KeyboardInterrupt. Setting interrupt event and breaking out of infinite loop...", flush=True)
            print("main: You may have to wait a minute for all child processes to gracefully exit!", flush=True)
            interrupt_event.set()
            break

    for proc in procs:
        print(f"main: Joining {proc.name}")
        proc.join()