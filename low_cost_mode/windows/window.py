from abc import ABC, abstractmethod
from pathlib import Path
from re import L
from typing import Tuple

import cv2
import numpy as np

from writer import ThreadedVideoWriter


class Window(ABC):
    def __init__(
        self,
        name: str,
        show: dict,
        write: dict,
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
        self._output_ext = output_ext
        self._queue_size = queue_size
        self._flush_thresh = flush_thresh
        self._fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._fps = fps
        self._frame_size = frame_size
        # Bit weird, but we pass in the same dict of `write`, `show` to all Windows,
        # and they can just pick their own name out of it
        if self._name not in show:
            print(f"Didn't find `show` for '{name}', defaulting to True")
        if self._name not in write:
            print(f"Didn't find `write` for '{name}', defaulting to False")
        self._show = show.get(self._name, True)
        self._write = write.get(self._name, False)

        # Creating cv2 window and/or Writer if needed
        if self._show:
            cv2.namedWindow(name)
        
        self._frame = None
        self._output_file = None
        self._writer = None

    def show(self):
        # Use if statement so we don't have to in App?
        if self._show:
            cv2.imshow(self._name, self._frame)

    def prepare_writer(self, output_dir: str):
        if not self._write:
            print(f"{str(self)} ignoring instruction to prepare writer, since self._write is False")
            return

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
        # Use if statement so we don't have to in App?
        if self._write:
            self._writer.write(self._frame)

    @abstractmethod
    def update(self, frame: np.ndarray):
        pass

    def release(self):
        if self._show:
            cv2.destroyWindow(self._name)
        if self._write:
            self._writer.release()

    def __str__(self) -> str:
        return f"Window: {self._name}"

    def __repr__(self) -> str:
        return f"<{str(self)}; show={self._show}; write={self._write}>"