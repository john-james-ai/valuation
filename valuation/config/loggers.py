#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/config/loggers.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Tuesday October 14th 2025 09:40:40 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import sys

from loguru import logger

from valuation.config.filepaths import LOGS_DATASET

# ------------------------------------------------------------------------------------------------ #


def configure_logging():
    """Configures logging for the application."""

    # Define module names for filtering logs
    # PROFILE_MODULE = "valuation.dataset.profile"
    # SALES_MODULE = "valuation.dataset.sales"
    # SPLIT_MODULE = "valuation.dataset.split"

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
