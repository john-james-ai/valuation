#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/core/state.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 16th 2025 08:04:32 pm                                              #
# Modified   : Wednesday October 22nd 2025 09:34:10 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Module defining workflow states."""
from enum import Enum

# ------------------------------------------------------------------------------------------------ #


class Status(Enum):
    """Enumeration of possible task statuses."""

    NOTSTARTED = ("Not Started", 0)
    RUNNING = ("Running", 1)
    SKIPPED = ("Skipped - Endpoint Exists", 2)
    WARNING = ("Warning", 3)
    SUCCESS = ("Success", 4)
    FAIL = ("Fail", 5)
