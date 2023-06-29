from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Event

import cv2

from config import Config, LOGGER
from motion_detector import MotionDetector
from reader import Reader
from writer import Writer


def main_loop(config: Config, output_directory: Path):
    # Create queues for transferring data between threads (or processes)
    reading_queue = Queue(maxsize=512)
    motion_input_queue = Queue(maxsize=512)
    writing_queue = Queue(maxsize=512)

    stop_signal = Event()

    video_filepath = output_directory / "motion001.avi"

    # Create all of our threads
    threads = (
        reader := Reader(
            reading_queue=reading_queue,
            video_source=config.video_source,
            stop_signal=stop_signal,
            sleep_seconds=config.reader_sleep_seconds,
            flush_proportion=config.reader_flush_proportion,
            logger=LOGGER,
        ),
        motion_detector := MotionDetector(
            input_queue=reading_queue,
            writing_queue=writing_queue,
            downscale_factor=config.downscale_factor,
            dilate_kernel_size=config.dilate_kernel_size,
            movement_threshold=config.movement_threshold,
            persist_factor=config.persist_factor,
            stop_signal=stop_signal,
            logger=LOGGER,
        ),
        writer := Writer.from_reader(
            reader=reader,
            writing_queue=writing_queue,
            filepath=video_filepath,
            stop_signal=stop_signal,
            logger=LOGGER,
        ),
    )

    for thread in threads:
        LOGGER.info(f"Starting {thread.name}")
        thread.start()

    # Regularly poll to check if all threads have finished. If they haven't finished,
    # just sleep a little and check later
    while True:
        try:
            time.sleep(5)
            if not any([thread.is_alive() for thread in threads]):
                LOGGER.info(
                    "All child processes appear to have finished! Exiting infinite loop..."
                )
                break

            for queue, queue_name in zip(
                [reading_queue, motion_input_queue, writing_queue],
                ["Reading", "Motion", "Writing"],
            ):
                LOGGER.debug(f"{queue_name} queue size: {queue.qsize()}")
        except (KeyboardInterrupt, Exception) as e:
            LOGGER.exception(
                "Received KeyboardInterrupt or some kind of Exception. Setting interrupt event and breaking out of infinite loop...",
            )
            LOGGER.warning(
                "You may have to wait a minute for all child processes to gracefully exit!",
            )
            stop_signal.set()
            break

    for thread in threads:
        LOGGER.info(f"Joining {thread.name}")
        thread.join()


def prepare_logging_handlers(output_directory: Path):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(threadName)-14s] %(msg)s"))
    file_handler = logging.FileHandler(filename=output_directory / "output.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)-8s] [%(threadName)-14s] %(msg)s")
    )
    # Make sure any prior handlers are removed
    LOGGER.handlers.clear()
    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)


def main(config: Config):
    # Figure out output filepath for this particular run
    output_directory = Path(f"out/{datetime.now()}")
    output_directory.mkdir()

    # Create some handlers for logging output to both console and file
    prepare_logging_handlers(output_directory=output_directory)
    
    # Save start time to measure duration at end of run
    start = time.time()

    LOGGER.info("Running main() with Config: ", config)
    LOGGER.info(f"Outputting to {output_directory}")

    # Make sure opencv doesn't use too many threads and hog CPUs
    cv2.setNumThreads(config.num_opencv_threads)

    # Copy config for record-keeping
    config.to_json_file(output_directory / "config.json")

    # Run main loop for this config     
    main_loop(config=config, output_directory=output_directory)

    # Add any extra stats/metadata to output too
    end = time.time()
    duration_seconds = end - start
    stats = {"duration_seconds": round(duration_seconds, 2)}
    with open(output_directory / "stats.json", "w") as f:
        json.dump(stats, f)

    LOGGER.info(f"Finished main() in {duration_seconds:.2f} seconds.")


if __name__ == "__main__":
    configs = Config.from_json_file_many(Path("config.json"))
    for config in configs:
        main(config=config)