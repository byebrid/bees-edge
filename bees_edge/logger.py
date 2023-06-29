from datetime import datetime
import logging
from pathlib import Path

def get_logger(output_directory: Path = None) -> logging.Logger:
    LOGGER = logging.getLogger("BeesEdgeLogger")
    LOGGER.setLevel(logging.DEBUG)

    if output_directory is None:
        output_directory = Path(f"out/{datetime.now()}")
        output_directory.mkdir(exist_ok=True)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(threadName)-14s] %(msg)s"))
    file_handler = logging.FileHandler(filename=output_directory / "output.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)-8s] [%(threadName)-14s] %(msg)s")
    )
    # Make sure any prior handlers are removed
    LOGGER.handlers.clear()
    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)

    return LOGGER