#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/config/loggers.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Saturday October 18th 2025 10:33:32 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from pathlib import Path
import sys

from loguru import logger

# ------------------------------------------------------------------------------------------------ #
# --- 1. Directories and Filepaths ---
PROJ_ROOT = Path(__file__).resolve().parents[2]


# LOG FILES
LOGS_DIR = PROJ_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure logs directory exists
LOGS_DATASET = LOGS_DIR / "dataset.log"
LOGS_MODELING = LOGS_DIR / "modeling.log"
LOGS_VALUATION = LOGS_DIR / "valuation.log"


def configure_logging():
    """Configures logging for the application."""

    try:
        logger.remove()  # Remove all previously added handlers

        # Configure a specific sink for dataset processing logs
        logger.add(
            LOGS_DATASET,
            level="DEBUG",
            # filter=lambda record: record["name"] in [PROFILE_MODULE, SALES_MODULE, SPLIT_MODULE],
            rotation="00:00",  # Rotate logs daily at midnight
            retention="7 days",  # Retain logs for 7 days
        )

        # Configure log to console.
        logger.add(
            sys.stderr,
            level="INFO",
            format="{message}",
            colorize=True,
        )

    except ModuleNotFoundError:
        pass
