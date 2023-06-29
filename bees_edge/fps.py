import numpy as np
from datetime import datetime as dt


class FPS:
    """
    Simple class to help determine FPS of some loop. It stores a circular buffer
    of frame durations, and the average FPS over the last `n` frames can be found
    via `FPS.get_average(self)`.

    Example usage
    -------------
    >>> from datetime import datetime as dt
    >>> fps = FPS(size=64)
    >>> while True:
    >>>     fps.tick()
    >>>     print(fps.get_average()) # Will be None after first tick, but will be actual FPS after other ticks!
    >>>     # Do some work ...
    """

    def __init__(self, size: int = 32) -> None:
        """
        Initialise the FPS object with a particular `size` of frame buffer.

        :param size: size of buffer array; i.e. you will get average FPS for last `size` frames, defaults to 32
        """
        self.size = size
        self.durations = np.zeros(
            size
        )  # array to store duration of each frame. We treat this as a circular buffer
        self.pointer = 0
        self.filled = False
        self.time = None

    def _append_duration(self, duration: float) -> None:
        """
        Internal method used to "append" a duration to the buffer. Note we say
        "append" because it is in reality a circular buffer, so once we hit end
        of array, we store next element back at the start of the array.

        :param duration: duration of frame in seconds
        """
        self.durations[self.pointer] = duration

        self.pointer += 1
        if self.pointer >= self.size:
            self.pointer = 0
            self.filled = True

    def tick(self):
        """
        Tell this FPS object to "tick" its clock. The time between ticks is the
        frame duration, and this method handles those calculations for you. Just
        call this at the start of every iterative loop.

        Note that for the first call of this method, there will be no frame
        duration calculated, since you need at least two ticks to calculate that!
        """
        if self.time is not None:
            now = dt.now()
            time_delta = now - self.time
            seconds = time_delta.total_seconds()
            self._append_duration(seconds)

        self.time = dt.now()

    def get_average(self) -> float:
        """
        Gets the average FPS based on however many frames have been processed so
        far. I.e. if only 5 ticks have been made, then this will be the average
        of the 4 corresponding frame durations, and so on up until the buffer has
        been completely filled.

        :return: Average FPS across last `n` frames, or `None` if no frames have
            been processed yet (i.e. if `tick()` has been called fewer than 2
            times).
        """
        max_index = self.size if self.filled else self.pointer
        # Means we haven't even had a single frame go by
        if max_index == 0:
            return None

        ave_duration = np.mean(self.durations[:max_index])
        ave_fps = 1 / ave_duration
        return ave_fps
