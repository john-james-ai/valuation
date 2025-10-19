#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/asset/type.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 18th 2025 08:09:04 pm                                              #
# Modified   : Saturday October 18th 2025 08:09:30 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from enum import StrEnum

# ------------------------------------------------------------------------------------------------ #


class AssetType(StrEnum):
    """Defines the recognized types of Entities."""

    DATASET = "dataset"
    MODEL = "model"
    REPORT = "report"
    PLOT = "plot"

    def __str__(self):
        return self.value
