from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Event

import cv2
import yappi

# Hate doing this, but simplest way to let python import sibling directories without
# actually installing this repo as a package
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bees_edge.config import Config
from bees_edge.logger import get_logger
from bees_edge.motion_detector import MotionDetector
from bees_edge.reader import Reader
from bees_edge.writer import Writer


def main_loop(config: Config, output_directory: Path, LOGGER: logging.Logger):
    # Create queues for transferring data between threads (or processes)
    reading_queue = Queue(maxsize=config.reading_queue_size)
    motion_input_queue = Queue(maxsize=config.motion_queue_size)
    writing_queue = Queue(maxsize=config.writing_queue_size)

    stop_signal = Event()

    video_filepath = output_directory / "motion001.avi"

    # Create all of our threads
    threads = (
        reader := Reader(
            reading_queue=reading_queue,
            video_source=config.video_source,
            stop_signal=stop_signal,
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
            sleeping_writer=config.sleeping_writer
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


def main(config: Config):
    yappi.start()

    # Figure out output filepath for this particular run
    output_directory = Path(f"out/{datetime.now()}")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Create some handlers for logging output to both console and file
    LOGGER = get_logger(output_directory=output_directory)
    
    # Save start time to measure duration at end of run
    start = time.time()

    LOGGER.info(f"Running main() with Config: {config}")
    LOGGER.info(f"Outputting to {output_directory}")

    # Make sure opencv doesn't use too many threads and hog CPUs
    cv2.setNumThreads(config.num_opencv_threads)

    # Copy config for record-keeping
    config.to_json_file(output_directory / "config.json")

    # Run main loop for this config     
    main_loop(config=config, output_directory=output_directory, LOGGER=LOGGER)

    # Add any extra stats/metadata to output too
    end = time.time()
    duration_seconds = end - start
    stats = {"duration_seconds": round(duration_seconds, 2)}
    with open(output_directory / "stats.json", "w") as f:
        json.dump(stats, f)

    LOGGER.info(f"Finished main() in {duration_seconds:.2f} seconds.")

    yappi.stop()

    # Log profiling stats
    LOGGER.info(f"Preparing profile...")
    threads = yappi.get_thread_stats()

    profile_filepath = output_directory / "profile.txt"
    with open(profile_filepath, "w") as f:
        for thread in threads:
            print(f"\n\nFunction stats for ({thread.name}) ({thread.id})", file=f)
            # LOGGER.debug(f"Function stats for ({thread.name}) ({thread.id})")
            yappi.get_func_stats(ctx_id=thread.id).print_all(out=f)
    
    LOGGER.info(f"Profile saved to {profile_filepath}")


if __name__ == "__main__":
    yappi.set_clock_type("cpu")
    
    configs = Config.from_json_file_many(Path("config.json"))
    for config in configs:
        main(config=config)
