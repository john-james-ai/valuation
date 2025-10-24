#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/modeling/model_selection/performance.py                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 24th 2025 12:37:17 am                                                #
# Modified   : Friday October 24th 2025 01:53:42 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Defines performance metrics for model evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from valuation.core.dataclass import DataClass
from valuation.utils.metrics import compute_smape, compute_wape

# ------------------------------------------------------------------------------------------------ #


@dataclass
class PerformanceMetrics(DataClass):
    model: str  # Model name
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    smape: float  # Symmetric Mean Absolute Percentage Error
    wape: float  # Weighted Absolute Percentage Error
    y_bar: float  # Mean of actual values
    n: int  # Number of observations

    @classmethod
    def compute(cls, model: str, y_true: npt.NDArray, y_pred: npt.NDArray) -> PerformanceMetrics:
        performance = {
            "model": model,
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,  # As percentage
            "smape": compute_smape(y_true, y_pred),
            "wape": compute_wape(y_true, y_pred),
            "y_bar": y_true.mean(),
            "n": len(y_true),
        }

        return cls(**performance)
