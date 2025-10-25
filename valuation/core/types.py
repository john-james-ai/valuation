#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/core/types.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 18th 2025 08:09:04 pm                                              #
# Modified   : Saturday October 25th 2025 10:22:21 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Defines the recognized types of assets."""
from enum import StrEnum

# ------------------------------------------------------------------------------------------------ #


class AssetType(StrEnum):
    """Defines the recognized types of Entities."""

    DATASET = "dataset"
    MODEL = "model"
    REPORT = "report"
    PLOT = "plot"
    ARTIFACT = "artifact"
    TABLE = "table"
