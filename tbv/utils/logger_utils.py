"""
Copyright (c) 2021
Argo AI, LLC, All Rights Reserved.

Notice: All information contained herein is, and remains the property
of Argo AI. The intellectual and technical concepts contained herein
are proprietary to Argo AI, LLC and may be covered by U.S. and Foreign
Patents, patents in process, and are protected by trade secret or
copyright law. This work is licensed under a CC BY-NC-SA 4.0 
International License.

Originating Authors: John Lambert
"""

import logging
from pathlib import Path

import argoverse.utils.datetime_utils as datetime_utils

import tbv.utils.dir_utils as dir_utils


LOGGING_DIR = Path(__file__).resolve().parent.parent.parent / "logging_output"


def get_logger() -> logging.Logger:
    """Set up a Python logger.

    Reference: https://github.com/mseg-dataset/mseg-semantic/blob/master/mseg_semantic/utils/logger_utils.py
    """
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger


def setup_logging() -> None:
    """Set up a Python logger that writes a log file to disk."""
    date_str = datetime_utils.generate_datetime_string()
    log_output_fpath = LOGGING_DIR / f"tbv_rendering_program_{date_str}.log"
    LOGGING_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Log will be saved to {log_output_fpath}")

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(filename)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename=log_output_fpath,
        level=logging.INFO,
    )
    logging.getLogger("boto").setLevel(logging.CRITICAL)
    logging.debug("Init Debug")
    logging.info("Init Info")
    logging.warning("Init Warning")
    logging.critical("Init Critical")
