from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from writer import ThreadedVideoWriter


class Window(ABC):
    def __init__(
        self,
        name: str,
        show: bool,
        write: bool,
        output_ext: str,
        queue_size: int,
        flush_thresh: int,
        fourcc: int,
        fps: int,
        frame_size: Tuple[int, int],
        **kwargs,
    ):
        # Just storing all initial parameters
        self._name = name
        self._show = show
        self._write = write
        self._output_ext = output_ext
        self._queue_size = queue_size
        self._flush_thresh = flush_thresh
        self._fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._fps = fps
        self._frame_size = frame_size

        # Creating cv2 window and/or Writer if needed
        if show:
            cv2.namedWindow(name)

        self._frame = None
        self._output_file = None
        self._writer = None

    def show(self):
        cv2.imshow(self._name, self._frame)

    def prepare_writer(self, output_dir: str):
        output_file = Path(output_dir) / self._name
        output_file = str(output_file.with_suffix(self._output_ext))
        self._output_file = output_file

        self._output_file = output_file
        self._writer = ThreadedVideoWriter(
            queue_size=self._queue_size,
            flush_thresh=self._flush_thresh,
            filename=output_file,
            fourcc=self._fourcc,
            fps=self._fps,
            frameSize=self._frame_size,
        )

    def write(self):
        self._writer.write(self._frame)

    @abstractmethod
    def update(self, frame: np.ndarray):
        pass

    def release(self):
        if self._show:
            cv2.destroyWindow(self._name)
        if self._write:
            self._writer.release()
