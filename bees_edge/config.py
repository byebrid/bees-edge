from __future__ import annotations
from itertools import product

import json
import logging
from pathlib import Path


# LOGGER = logging.getLogger("BeesEdgeLogger")
# LOGGER.setLevel(logging.DEBUG)


class Config:
    """Configuration class just to provide better parameter hints in code editor."""
    
    # These keys are naturally list-like (e.g. have 2 elements, width & height), 
    # so flag them as such. Important when converting a single JSON file to many
    # Config objects, since that generates combinations of Configs based on which
    # parameter values are (unnaturally) lists or not.
    # We assume we'll only ever need *one*-level lists, no further nesting
    _NATURALLY_LISTS = (
        "hires_size"
        "lores_size"
    )

    def __init__(
        self,
        frame_guarantee_interval: int,
        hires_size: int,
        lores_size: int,
        threshold: int,
        min_moving_pixels: int,
        queue_size: int,
        output_fps: int,
        timeout_seconds: int,
        out_root_directory: str,
        dilation_size: int
    ) -> None:
        self._d = {
            "frame_guarantee_interval": frame_guarantee_interval,
            "hires_size": hires_size,
            "lores_size": lores_size,
            "threshold": threshold,
            "min_moving_pixels": min_moving_pixels,
            "queue_size": queue_size,
            "output_fps": output_fps,
            "timeout_seconds": timeout_seconds,
            "out_root_directory": out_root_directory,
            "dilation_size": dilation_size
        } 

    def __getattr__(self, name: str):
        return self._d[name]

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
            if not isinstance(value, list):
                config_dict[key] = [value]
            elif key in Config._NATURALLY_LISTS:
                # Check for nested list in this case
                if not isinstance(value[0], list):
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