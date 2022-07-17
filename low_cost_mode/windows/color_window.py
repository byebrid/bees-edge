from typing import Tuple

import cv2
import numpy as np
from utils import refine_mask

from windows.window import Window


class ColorWindow(Window):
    NAME = "Color"

    def __init__(
        self,
        HSV_dict: dict,
        show: bool,
        write: bool,
        output_ext: str,
        queue_size: int,
        flush_thresh: int,
        fourcc: int,
        fps: int,
        frame_size: Tuple[int, int],
        thresh: int,
        dilate: int,
        pad: int,
        min_area: int,
        **kwargs
    ):
        name = self.NAME
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

        self._thresh = thresh
        self._dilate = dilate
        self._pad = pad
        self._min_area = min_area

        self._HSV_DICT = HSV_dict
        self._low_H = HSV_dict["LOW"]["H"]
        self._low_S = HSV_dict["LOW"]["S"]
        self._low_V = HSV_dict["LOW"]["V"]
        self._high_H = HSV_dict["HIGH"]["H"]
        self._high_S = HSV_dict["HIGH"]["S"]
        self._high_V = HSV_dict["HIGH"]["V"]

    def update(self, frame: np.ndarray, index: int):
        # Convert original frame to HSV since color thresholding is more sensible
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Find all pixels within given HSV range
        hsv_low = (self._low_H, self._low_S, self._low_V)
        hsv_high = (self._high_H, self._high_S, self._high_V)
        inrange = cv2.inRange(hsv, hsv_low, hsv_high)

        # Get mask from colored pixels, and apply mask to original frame
        # Note that we pass in color=None, since we already have gray data
        color_frame = refine_mask(
            frame=frame,
            mask=inrange,
            ret_mask=False,
            thresh=self._thresh,
            pad=self._pad,
            min_area=self._min_area,
            color=None,
        )

        self._frame = color_frame
