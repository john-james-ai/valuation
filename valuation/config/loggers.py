#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mercor Dominick's Fine Foods Acquisition Analysis                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/config/loggers.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mercor-dominicks-acquisition-analysis              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Saturday October 11th 2025 01:23:02 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import sys

from loguru import logger

LOG_MESSAGES_RECORDED = []


# ------------------------------------------------------------------------------------------------ #
# Configure logging
def spy_sink(message):
    """A simple sink that just records messages to our global list."""
    LOG_MESSAGES_RECORDED.append(message.strip())


def configure_logging():
    """Configures logging for the application."""

    # Define module names for filtering logs
    PROFILE_MODULE = "valuation.dataset.profile"
    SALES_MODULE = "valuation.dataset.sales"
    SPLIT_MODULE = "valuation.dataset.split"

    try:
        logger.remove()  # Remove all previously added handlers

        # # Configure a specific sink for dataset processing logs
        # logger.add(
        #     LOGS_DATASET,
        #     level="DEBUG",
        #     # filter=lambda record: record["name"] in [PROFILE_MODULE, SALES_MODULE, SPLIT_MODULE],
        #     rotation="1 week",
        # )

        # Configure log to console.
        logger.add(
            sys.stderr,
            level="INFO",
            format="{message}",
            colorize=True,
        )

        logger.add(spy_sink, level="INFO")
    except ModuleNotFoundError:
        pass
