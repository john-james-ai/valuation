#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/core/stage.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 18th 2025 08:07:16 pm                                              #
# Modified   : Wednesday October 22nd 2025 06:54:24 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Defines the recognized stages of asset development."""
from enum import StrEnum

# ------------------------------------------------------------------------------------------------ #


class Stage(StrEnum):
    """Defines the recognized types of Entities."""


class DatasetStage(Stage):

    RAW = "raw"
    INGEST = "ingest"
    CLEAN = "clean"
    TRANSFORM = "transform"
    FEATURES = "feature_engineered"
    ENRICHED = "enriched"
    EXTERNAL = "external"
    REFERENCE = "reference"
    FINAL = "final"
    TEST = "test"

    def __str__(self):
        return self.value


class ModelStage(Stage):
    INITIAL = "initial"
    TUNED = "tuned"
    FINAL = "final"

    def __str__(self):
        return self.value
