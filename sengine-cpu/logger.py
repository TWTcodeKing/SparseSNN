"""Logging for sengine-cpu."""

import logging

def get_logger(name: str = "sengine-cpu") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s  [%(name)s:%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

log = get_logger()
