from typing import Tuple
from windows.window import Window
import cv2
import numpy as np
from utils import refine_mask, detect_motion, detect_motion_old_way


class MovementWindow(Window):
    NAME = "Movement"

    def __init__(
        self,
        thresh: int,
        dilate: int,
        pad: int,
        min_area: int,
        show: bool,
        write: bool,
        output_ext: str,
        queue_size: int,
        flush_thresh: int,
        fourcc: int,
        fps: int,
        frame_size: Tuple[int, int],
        **kwargs
    ):
        name = self.NAME
        self._thresh = thresh
        self._dilate = dilate
        self._pad = pad
        self._min_area = min_area

        super().__init__(
            name=name,
            show=show,
            write=write,
            output_ext=output_ext,
            queue_size=queue_size,
            flush_thresh=flush_thresh,
            fourcc=fourcc,
            fps=fps,
            frame_size=frame_size,
            **kwargs
        )

        self._prev_frame = None

        

    def update(self, frame: np.ndarray, index: int):
        if self._prev_frame is None:
            self._prev_frame = frame

        # Get mask based on movement
        motion_mask = detect_motion(frame=frame, prev_frame=self._prev_frame)
        # motion_mask = detect_motion_old_way(frame=frame, prev_frame=self._prev_frame, thresh=40, pad=0)

        # Create copy of original frame that only includes moving parts
        diff_frame = np.zeros(frame.shape, dtype=np.uint8)
        diff_frame[motion_mask] = frame[motion_mask]

        self._prev_frame = frame
        self._frame = diff_frame
        self._mask = motion_mask
