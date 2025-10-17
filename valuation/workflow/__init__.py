#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/workflow/__init__.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 16th 2025 08:04:32 pm                                              #
# Modified   : Friday October 17th 2025 03:25:24 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Workflow package."""
from enum import Enum

# ------------------------------------------------------------------------------------------------ #


class Status(Enum):
    """Enumeration of possible task statuses."""

    PENDING = "Pending"
    SUCCESS = "Success"
    FAILURE = "Failure"
    CRITICAL = "Critical Failure"
    WARNING = "Warning"
    EXISTS = "Output Exists"
