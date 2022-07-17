from typing import Tuple

import numpy as np
from windows.window import Window


class TrailWindow(Window):
    NAME = "Trail"

    def __init__(self, show: dict, write: dict, output_ext: str, queue_size: int, flush_thresh: int, fourcc: int, fps: int, frame_size: Tuple[int, int], in_window: Window = None, **kwargs):
        if in_window is None:
            raise ValueError(f"{str(self)} needs input from MovementWindow to function properly!")

        name=self.NAME
        super().__init__(name=name, show=show, write=write, output_ext=output_ext, queue_size=queue_size, flush_thresh=flush_thresh, fourcc=fourcc, fps=fps, frame_size=frame_size, in_window=in_window, **kwargs)

    def update(self, frame: np.ndarray, index: int):
        # We're expecting to use movement frame as our input window here
        diff_frame = self._in_window._frame
        mask = self._in_window._mask

        if self._frame is None:
            self._frame = diff_frame.copy()
        else:
            # Double-check that none of these pixels have been updated before
            if not np.any(self._frame, where=mask):
                self._frame[mask] = frame[mask]