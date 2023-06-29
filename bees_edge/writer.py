from __future__ import annotations

from logging import Logger
from queue import Empty, Queue
from threading import Event
import time
from typing import Tuple

import cv2

from bees_edge.logging_thread import LoggingThread
from bees_edge.reader import Reader


class Writer(LoggingThread):
    def __init__(
        self,
        writing_queue: Queue,
        filepath: str,
        frame_size: Tuple[int, int],
        fps: int,
        stop_signal: Event,
        logger: Logger,
        sleep_seconds: float = 0.1,
    ) -> None:
        super().__init__(name="WriterThread", logger=logger)

        self.writing_queue = writing_queue
        self.filepath = filepath
        self.frame_size = frame_size
        self.fps = fps
        self.stop_signal = stop_signal

        # For smart sleep every iteration
        self.sleep_seconds = sleep_seconds

        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.flush_thresh = int(0.75 * writing_queue.maxsize)
        self.info(
            f"Will flush buffer to output file every {self.flush_thresh} frames"
        )

        self.frame_count = 0

    @classmethod
    def from_reader(
        cls,
        reader: Reader,
        writing_queue: Queue,
        filepath: str,
        stop_signal: Event,
        logger: Logger
    ) -> Writer:
        """Convenience method to generate a Writer from a Reader.

        This is useful because the Writer should share the FPS and resolution
        of the input video as determined by the Reader. This just saves you
        having to parse those attributes yourself.

        Parameters
        ----------
        reader : Reader
            Reader whose 
        writing_queue : Queue
            Queue to retrieve video frames from. Some other thread should be 
            putting these frames into this queue for this Writer to retrieve.
        filepath : str
            Filepath for output video file.
        stop_signal : Event
            A threading Event that this Reader queries to know when to stop. 
            This is used for graceful termination of the multithreaded program.
        logger : logging.Logger
            Logger to use for logging key info, warnings, etc.

        Returns
        -------
        Writer
            Writer with same FPS and frame size as given Reader.
        """
        fps = reader.get_fps()
        frame_size = reader.get_frame_size()
        writer = Writer(
            writing_queue=writing_queue,
            filepath=filepath,
            frame_size=frame_size,
            fps=fps,
            stop_signal=stop_signal,
            logger=logger,
        )
        return writer

    def run(self) -> None:
        vw = cv2.VideoWriter(
            filename=str(self.filepath), # Ensure string instead of Path object
            fourcc=self.fourcc,
            fps=self.fps,
            frameSize=self.frame_size,
        )
        
        omitted_frames = []
        currently_omitting = False

        loop_is_running = True
        while loop_is_running:
            self.smart_sleep()

            # Only flush the threshold number of frames, OR remaining frames if there are only a few left
            flush_thresh = int(0.65 * self.writing_queue.maxsize)
            frames_to_flush = min(self.writing_queue.qsize(), flush_thresh)
            self.debug(f"Flushing {frames_to_flush} frames...")

            for i in range(frames_to_flush):
                try:
                    frame = self.writing_queue.get(timeout=10)
                except Empty:
                    self.warning(f"Waited too long for frame! Exiting...")
                    loop_is_running = False
                    break
                if frame is None:
                    loop_is_running = False
                    break

                vw.write(frame)
                # TODO: Re-add the frame omission, only commenting it out for downscaling experiment
                # # Ignore frames that have *zero* movement in them
                # if np.any(frame):
                #     vw.write(frame)
                #     if currently_omitting:
                #         # Append *end* of all-black interval to list
                #         currently_omitting = False
                #         omitted_frames.append(self.frame_count)
                # elif not currently_omitting:
                #     # Append *start* of all-black interval to list
                #     currently_omitting = True
                #     omitted_frames.append(self.frame_count)
                
                self.frame_count += 1

                if self.frame_count % 1000 == 0:
                    self.info(f"Written {self.frame_count} frames so far")
            self.debug(f"Flushed {frames_to_flush} frames!")

        vw.release()

        # TODO: Re-add the frame omission, only commenting it out for downscaling experiment
        # # Write CSV file with omitted frame indices. Note this is not the most
        # # space-efficient way to store these, but it's probs good enough
        # output_dir = Path(self.filepath).parent
        # csv_filepath = output_dir / "omitted_frames.csv"
        # with open(csv_filepath, "w") as f:
        #     csv_writer = csv.writer(f)
        #     csv_writer.writerow(omitted_frames)

    def smart_sleep(self):
        time.sleep(self.sleep_seconds)

        # If we're stopping, don't change sleep time since we want to flush whatever's
        # left in queue, which may not be very many frames (and we don't want to
        # extend sleep time at very end of program!)
        if self.stop_signal.is_set():
            return

        # Define boundaries for very bad, sort of bad, and good queue sizes
        very_lower = int(0.10 * self.writing_queue.maxsize)
        lower = int(0.25 * self.writing_queue.maxsize)
        upper = int(0.75 * self.writing_queue.maxsize)
        very_upper = int(0.90 * self.writing_queue.maxsize)
        
        # Multipliers to extend/shorten sleep if queue not in sweet zone
        sleep_longer_multiplier = 1.05
        sleep_shorter_multiplier = 0.93 # Intentionally not exact reciprocal so we can get arbitrary sleep time
        # Extra multiplier for extreme edge cases
        exaggerate_factor = 1.1

        qsize = self.writing_queue.qsize()

        if lower <= qsize <= upper:
            return
        
        if qsize < lower:
            multiplier = sleep_longer_multiplier

            if qsize < very_lower:
                multiplier *= exaggerate_factor
                self.debug(f"Exaggerating sleep duration multiplier by x{exaggerate_factor} due to *very* empty queue")    
        elif qsize > upper:
            multiplier = sleep_shorter_multiplier

            if qsize > very_upper:
                multiplier /= exaggerate_factor
                self.debug(f"Exaggerating sleep duration multiplier by ÷{exaggerate_factor} due to *very* full queue")    
            
        new_sleep = self.sleep_seconds * multiplier
        self.debug(f"Sleep = {self.sleep_seconds:.4f} -> {new_sleep:.4f} (x{multiplier:.3f})")
        self.sleep_seconds = new_sleep
