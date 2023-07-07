from __future__ import annotations
from itertools import product

import json
import logging
from pathlib import Path


# LOGGER = logging.getLogger("BeesEdgeLogger")
# LOGGER.setLevel(logging.DEBUG)


class Config:
    """Configuration class just to provide better parameter hints in code editor."""
    def __init__(
        self,
        video_source: str,
        downscale_factor: int,
        dilate_kernel_size: int,
        movement_threshold: int,
        persist_factor: float,
        num_opencv_threads: int,
        sleeping_writer: bool,
        reading_queue_size: int,
        motion_queue_size: int,
        writing_queue_size: int
    ) -> None:
        self.video_source = video_source
        self.downscale_factor = downscale_factor
        self.dilate_kernel_size = dilate_kernel_size
        self.movement_threshold = movement_threshold
        self.persist_factor = persist_factor
        self.num_opencv_threads = num_opencv_threads
        self.sleeping_writer = sleeping_writer
        self.reading_queue_size = reading_queue_size
        self.motion_queue_size = motion_queue_size
        self.writing_queue_size = writing_queue_size

    @staticmethod
    def from_json_file(filepath: Path) -> Config:
        """Create a Config instance from a JSON file.
        
        Note that the config file should only have *one* value per parameter. If
        you would like to test multiple combinations of parameter values, use
        `from_json_file_many()` instead.
        """
        with open(filepath, "r") as f:
            config_dict = json.load(f)
            return Config(**config_dict)
        
    @staticmethod
    def from_json_file_many(filepath: Path) -> list[Config]:
        """Create list of Config objects from a single JSON file.
        
        To use this, set parameter values to be *lists* of values in your config
        file. This method will then create a Config object from every possible 
        combination of parameter values in this file. For example, if you set
        2 values for parameter A, 3 for B, and 5 for C, you will end up with
        2*3*5=30 different Config objects.

        This is useful when performing parameter sweeps. This also works as a
        lazy way of repeating an experiment a couple of times, e.g. just by
        copying the same parameter value three times for one parameter to run
        three identical trials.

        WARNING: the config file is expected to be perfectly flat (no nesting),
        and a single value is never expected to be represented by a list. I.e.
        if you need the ability to use a list as a single parameter value, then
        you will need to modify this Config code!
        """
        with open(filepath, "r") as f:
            config_dict = json.load(f)

        for key, value in config_dict.items():
            # Cast every parameter to be a list, even if it's just a single element
            if type(value) is not list:
                config_dict[key] = [value]

        # Get every combination of parameter values now
        value_combos = product(*config_dict.values())

        # Now create Config objects from every one of these combos of parameter values
        configs = []
        for combo in value_combos:
            combo_dict = dict(zip(config_dict.keys(), combo))
            config = Config(**combo_dict)
            configs.append(config)

        return configs
    
    def to_json_file(self, filepath: Path):
        """Save this config to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.__dict__, f)

    def __str__(self) -> str:
        return f"<Config: {self.__dict__}>"
    
    def __repr__(self) -> str:
        return str(self)
