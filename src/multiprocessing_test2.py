from multiprocessing import Queue, Process
from random import randint
import time


def producer(queue: Queue, sleep: float = 0.2, num: int = 10):
    print("Producer: STARTED!", flush=True)
    try:
        for i in range(num):
            item = randint(0, 100)
            print(f"Producer: putting item#{i}={item} onto queue", flush=True)
            queue.put(item)
            time.sleep(sleep)
    except KeyboardInterrupt:
        print(f"Producer: RECEIVED KEYBOARDINTERRUPT. Stopping putting items onto producer queue...")
    finally:
        print("Producer: Putting None on end of queue")
        queue.put(None)
        print("Producer: FINISHED!", flush=True)


def consumer(queue: Queue, sleep: float=0.3):
    print("Consumer: STARTED!", flush=True)
    i = 0
    while True:
        try:
            print(f"Consumer: Waiting for item...", flush=True)
            item = queue.get()
            print(f"Consumer: Got item#{i}={item}", flush=True)
            time.sleep(sleep)
            if item is None:
                print("Consumer: Hit end of queue")
                break

            i += 1
        except KeyboardInterrupt:
            print(f"Consumer: RECEIVED KEYBOARDINTERRUPT. Continuing to empty from queue...", flush=True)
    print("Consumer: FINISHED!", flush=True)


if __name__ == "__main__":
    producer_queue = Queue(maxsize=512)
    consumer_queue = Queue(maxsize=512)

    p1 = Process(target=producer, args=(producer_queue,))
    p2 = Process(target=consumer, args=(consumer_queue,))

    p2.start()
    p1.start()

    while True:
        try:
            item = producer_queue.get()
            consumer_queue.put(item)

            if item is None:
                break
        except KeyboardInterrupt:
            print("main: RECEIVED KEYBOARDINTERRUPT. Continuing to move items from producer queue onto consumer queue...")

    p1.join()
    p2.join()

    print(f"Producer queue: {producer_queue.qsize()} items", flush=True)
    print(f"Consumer queue: {consumer_queue.qsize()} items", flush=True)

