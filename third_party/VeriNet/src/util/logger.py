
"""
Logger functionality

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import logging
import os

from verinet.util.config import CONFIG


def get_logger(level, name: str, directory: str, filename: str):
    
    """
    Returns a logger saving log to file

    Args:
        level:
            Severity of the logger
        name:
            Name of the logger
        directory:
            The directory of the log file
        filename:
            The name of the logfile
    Returns:
        The logger
    """

    if CONFIG.USE_LOGGER:

        if not os.path.isdir(directory):
            os.mkdir(directory)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if CONFIG.USE_LOGGER:
        handler = logging.FileHandler(directory + filename)
        handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if not CONFIG.USE_LOGGER or not CONFIG.LOGS_TERMINAL:
        logger.propagate = False

    return logger
