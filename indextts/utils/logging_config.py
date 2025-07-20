import logging
import sys

def setup_logging(level=logging.INFO):
    """
    Set up the logging configuration.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("indextts.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
