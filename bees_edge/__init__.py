# Hate doing this, but simplest way to let python import sibling directories without
# actually installing this repo as a package
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bees_edge import (app, config, fps, logging_thread, motion_detector, reader, writer)