#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/modeling/model_selection/cv.py                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 23rd 2025 11:52:39 pm                                              #
# Modified   : Friday October 24th 2025 12:27:49 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #


from dataclasses import dataclass

from valuation.core.dataclass import DataClass
from valuation.flow.modeling.model_selection.mlforecast import NUM_CORES

# ------------------------------------------------------------------------------------------------ #
N_JOBS = NUM_CORES - 2


# ------------------------------------------------------------------------------------------------ #
@dataclass
class CrossValidationHP(DataClass):
    h: int = 26  # Forecast horizon for each window
    n_windows: int = 3  # Initial training period
    step_size: int = 13  # Step size between windows
    fitted: bool = True  # Whether to return fitted values
    dropna: bool = True  # Whether to drop NA values

    def as_dict(self) -> dict:
        """Returns the hyperparameters as a dictionary."""
        return {
            "h": self.h,
            "n_windows": self.n_windows,
            "step_size": self.step_size,
            "fitted": self.fitted,
            "dropna": self.dropna,
        }
