"""
This is for testing whether sleeping the reader actually improves performance or 
not when reading from a video file.
"""
# Hate doing this, but simplest way to let python import sibling directories without
# actually installing this repo as a package
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bees_edge.config import Config
from bees_edge.app import main

# Total number of trials to run. Should be even so sleep and no-sleep get same
# number of trials each
NUM_TRIALS = 12

if __name__ == "__main__":
    # config = Config.from_json_file(Path("config.json"))
    config = Config(
        video_source="/home/lex/Development/bees-edge/data/strawberry/cam_1_N_video_20210315_132804.h264.avi",
        reader_sleep_seconds=6.7,
        reader_flush_proportion=0.9,
        downscale_factor=2,
        dilate_kernel_size=128,
        movement_threshold=40,
        persist_factor=0.65,
        num_opencv_threads=10,
        sleeping_disabled=False
    )
    
    # Alternate between sleeping and not sleeping just in case the program slows
    # down on average as time goes on, which could lead to misleading results if
    # we just tested sleeping 6 times, and *then* no-sleeping 6 times.
    for i in range(NUM_TRIALS):
        main(config)
        # Toggle between enabled/disabled every run
        config.sleeping_disabled = not config.sleeping_disabled