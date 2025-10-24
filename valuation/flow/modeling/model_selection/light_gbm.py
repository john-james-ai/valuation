#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/modeling/model_selection/light_gbm.py                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 23rd 2025 11:49:35 pm                                              #
# Modified   : Friday October 24th 2025 03:15:12 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Defines hyperparameters for LightGBM model."""
from typing import Any, Dict

from dataclasses import dataclass

from valuation.core.dataclass import DataClass
from valuation.flow.modeling.model_selection.mlforecast import NUM_CORES

# ------------------------------------------------------------------------------------------------ #
N_JOBS = NUM_CORES - 2


# ------------------------------------------------------------------------------------------------ #
@dataclass
class LightGBMHP(DataClass):
    verbosity: int = -1  # Suppress LightGBM output
    objective: str = "tweedie"  # Objective function
    tweedie_variance_power: float = 1.2  # Tweedie variance power
    n_estimators: int = 1000  # Number of trees for LightGBM
    learning_rate: float = 0.02  # Learning rate
    max_depth: int = 7  # Maximum tree depth
    num_leaves: int = 63  # Number of leaves in full trees
    min_child_samples: int = 20  # Minimum data in leaf
    subsample: float = 0.8  # Row subsampling
    colsample_bytree: float = 0.7  # Feature subsampling
    reg_alpha: float = 1.0  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    subsample_freq: int = 1  # Subsample every tree
    min_split_gain: float = 0.1  # Minimum split gain
    min_child_weight: float = 1.0  # Minimum child weight
    random_state: int = 42  # Fixed random seed
    n_jobs: int = N_JOBS  # Use all available cores - 2

    def as_dict(self) -> Dict[str, Any]:
        """Returns the hyperparameters as a dictionary."""
        return {
            "verbosity": self.verbosity,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "subsample_freq": self.subsample_freq,
            "min_split_gain": self.min_split_gain,
            "min_child_weight": self.min_child_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }
