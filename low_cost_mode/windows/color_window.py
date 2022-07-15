from typing import Tuple
from window import Window


class ColorWindow(Window):  
    NAME = "Color"

    def __init__(self, HSV_dict: dict, show: bool, write: bool, output_dir: str, queue_size:int, flush_thresh:int, fourcc: int, fps: int, frame_size: Tuple[int, int], **kwargs):
        name = self.NAME
        super().__init__(name=name, show=show, write=write, output_dir=output_dir, queue_size=queue_size, flush_thresh=flush_thresh, fourcc=fourcc, fps=fps, frame_size=frame_size, **kwargs)

        self._HSV_DICT = HSV_dict

    def update(self):
        pass