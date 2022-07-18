from typing import Tuple

import cv2
import numpy as np

from windows.window import Window


class InputWindow(Window):
    NAME = "Input"

    def __init__(self, show: bool, write: bool, output_ext: str, queue_size: int, flush_thresh: int, fourcc: int, fps: int, frame_size: Tuple[int, int], **kwargs):
        name = self.NAME
        
        super().__init__(name=name, show=show, write=write, output_ext=output_ext, queue_size=queue_size, flush_thresh=flush_thresh, fourcc=fourcc, fps=fps, frame_size=frame_size, **kwargs)

    def update(self, frame: np.ndarray, index: int):
        frame = frame.copy()
        cv2.putText(
            frame,
            f"Frame {index}",
            (50, 100),
            self.FONT,
            1,
            (0, 0, 255),
            3,
        )

        self._frame = frame

