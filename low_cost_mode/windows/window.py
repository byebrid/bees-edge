from abc import ABC, abstractmethod

import cv2


class Window(ABC):
    DEFAULT_FOURCC = "XVID"
    DEFAULT_EXT = ".avi"

    def __init__(
        self,
        *,
        name: str,
        show: bool = False,
        write: bool = False,
        output_dir: str = None,
        queue_size=100,
        flush_thresh=50,
        fourcc: int = None,
        fps: int = None,
        frame_size: Tuple[int, int] = None,
        **kwargs,
    ):
        """
        ! You must use keyword arguments with this initialiser !
        """
        if write and output_dir is None:
            raise ValueError("Can't write if no `output_dir` specified!")
        if (show or write) and (frame_size is None or fps is None):
            raise ValueError("Can't show or write if `frame_size` or `fps` is None!")
        if fourcc is None:
            print(f"No fourcc provided, assuming default '{self.DEFAULT_FOURCC}'")
            fourcc = cv2.VideoWriter_fourcc(*self.DEFAULT_FOURCC)

        # Just storing all initial parameters
        self._name = name
        self._show = show
        self._write = write
        self._output_dir = output_dir
        self._queue_size = queue_size
        self._flush_thresh = flush_thresh
        self._fourcc = fourcc
        self._fps = fps
        self._frame_size = frame_size

        # Creating cv2 window and/or Writer if needed
        if show:
            cv2.namedWindow(name)
        if write:
            output_file = Path(output_dir) / name
            output_file = str(output_file.with_suffix(self.DEFAULT_EXT))
            self._output_file = output_file
            self._writer = ThreadedVideoWriter(
                queue_size=queue_size,
                flush_thresh=flush_thresh,
                filename=output_file,
                fourcc=fourcc,
                fps=fps,
                frameSize=frame_size,
            )

        self._frame = None

    def show(self):
        cv2.imshow(self._name, self._frame)

    def write(self):
        self._writer.write(self._frame)

    @abstractmethod
    def update(self):
        pass
