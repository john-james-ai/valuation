#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/modeling/model_selection/mlforecast.py                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 23rd 2025 11:37:53 pm                                              #
# Modified   : Friday October 24th 2025 12:03:03 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Defines hyperparameters for MLForecast model."""
from dataclasses import dataclass, field

from mlforecast.lag_transforms import RollingMean

from valuation.core.dataclass import DataClass

# ------------------------------------------------------------------------------------------------ #
NUM_CORES = 24
NUM_THREADS = max(1, NUM_CORES - 2)  # Use all but two cores for training


@dataclass
class MLForecastHP(DataClass):
    freq: str = "W-WED"  # Weekly frequency ending on Wednesday
    lags: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    lag_transforms: dict[int, list] = field(
        default_factory=lambda: {1: [RollingMean(window_size=4)]}
    )
    num_threads: int = NUM_THREADS  # Use all but two cores for training]
    date_features: list[str] = field(
        default_factory=lambda: ["week", "month", "dayofyear"]
    )  # Date features to include
    target_transforms: list = field(default_factory=lambda: [])  # No target transforms

    def as_dict(self) -> dict:
        """Returns the hyperparameters as a dictionary."""
        return {
            "freq": self.freq,
            "lags": self.lags,
            "lag_transforms": self.lag_transforms,
            "num_threads": self.num_threads,
            "date_features": self.date_features,
            "target_transforms": self.target_transforms,
        }
